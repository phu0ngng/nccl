/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "info.h"
#include "bootstrap.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport,
  netTransport,
};

template <int type>
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connect, struct ncclConnector* connector, int channelId) {
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports+t;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    int ret = 0;
    NCCLCHECK(transport->canConnect(&ret, comm->topo, graph, myInfo, peerInfo));
    if (ret) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId));
      return ncclSuccess;
    }
  }
  WARN("No transport found !");
  return ncclInternalError;
}

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, struct ncclChannel* channel, int nrecv, int* peerRecv, int nsend, int* peerSend) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  uint32_t mask = 1 << channel->id;
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1 || peer >= comm->nRanks || channel->peers[peer].recv.connected) continue;
    comm->connectRecv[peer] |= mask;
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1 || peer >= comm->nRanks || channel->peers[peer].send.connected) continue;
    comm->connectSend[peer] |= mask;
  }
  return ncclSuccess;
}

ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph) {
  struct ncclConnect* connectDataSend, *connectDataRecv;
  NCCLCHECK(ncclCalloc(&connectDataSend, MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&connectDataRecv, MAXCHANNELS));
  for (int i=1; i<comm->nRanks; i++) {
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    int sendPeer = (comm->rank + i) % comm->nRanks;
    uint32_t recvMask = comm->connectRecv[recvPeer];
    uint32_t sendMask = comm->connectSend[sendPeer];
    int recvStartC = MAXCHANNELS, recvEndC = 0;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[recvPeer].recv;
        recvStartC = std::min(recvStartC, c);
        recvEndC = std::max(recvEndC, c+1);
        NCCLCHECK(selectTransport<0>(comm, graph, comm->peerInfo+comm->rank, comm->peerInfo+recvPeer, connectDataRecv+c, conn, c));
      }
    }
    int sendStartC = MAXCHANNELS, sendEndC = 0;
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[sendPeer].send;
        sendStartC = std::min(sendStartC, c);
        sendEndC = std::max(sendEndC, c+1);
        NCCLCHECK(selectTransport<1>(comm, graph, comm->peerInfo+comm->rank, comm->peerInfo+sendPeer, connectDataSend+c, conn, c));
      }
    }
    if (recvEndC) NCCLCHECK(bootstrapSend(comm->bootstrap, recvPeer, connectDataRecv+recvStartC, sizeof(struct ncclConnect)*(recvEndC-recvStartC)));
    if (sendEndC) NCCLCHECK(bootstrapSend(comm->bootstrap, sendPeer, connectDataSend+sendStartC, sizeof(struct ncclConnect)*(sendEndC-sendStartC)));
    if (sendEndC) NCCLCHECK(bootstrapRecv(comm->bootstrap, sendPeer, connectDataSend+sendStartC, sizeof(struct ncclConnect)*(sendEndC-sendStartC)));
    if (recvEndC) NCCLCHECK(bootstrapRecv(comm->bootstrap, recvPeer, connectDataRecv+recvStartC, sizeof(struct ncclConnect)*(recvEndC-recvStartC)));
    for (int c=0; c<MAXCHANNELS; c++) {
      if (sendMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[sendPeer].send;
        NCCLCHECK(conn->transportComm->connect(comm, connectDataSend+c, 1, comm->rank, conn));
        conn->connected = 1;
        CUDACHECK(cudaMemcpy(&comm->channels[c].devPeers[sendPeer].send, conn, sizeof(struct ncclConnector), cudaMemcpyHostToDevice));
      }
    }
    for (int c=0; c<MAXCHANNELS; c++) {
      if (recvMask & (1<<c)) {
        struct ncclConnector* conn = &comm->channels[c].peers[recvPeer].recv;
        NCCLCHECK(conn->transportComm->connect(comm, connectDataRecv+c, 1, comm->rank, conn));
        conn->connected = 1;
        CUDACHECK(cudaMemcpy(&comm->channels[c].devPeers[recvPeer].recv, conn, sizeof(struct ncclConnector), cudaMemcpyHostToDevice));
      }
    }
    comm->connectRecv[recvPeer] = comm->connectSend[sendPeer] = 0;
  }
  return ncclSuccess;
}

