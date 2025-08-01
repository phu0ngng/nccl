#include "core.h"
#include "comm.h"
#include "nccl_device/impl/core__funcs.h"

NCCL_API_CXX(ncclTeam, ncclTeamWorld, struct ncclComm* comm);
ncclTeam ncclTeamWorld(struct ncclComm* comm) {
  ncclTeam ans;
  ans.nRanks = comm->nRanks;
  ans.rank = comm->rank;
  ans.stride = 1;
  return ans;
}

NCCL_API_CXX(ncclTeam, ncclTeamLsa, struct ncclComm* comm);
ncclTeam ncclTeamLsa(struct ncclComm* comm) {
  ncclTeam ans;
  ans.nRanks = comm->symrState.lsaSize;
  ans.rank = comm->symrState.lsaSelf;
  ans.stride = 1;
  return ans;
}

NCCL_API_CXX(ncclTeam, ncclTeamRail, struct ncclComm* comm);
ncclTeam ncclTeamRail(struct ncclComm* comm) {
  ncclTeam ans;
  ans.nRanks = comm->nRanks/comm->symrState.lsaSize;
  ans.rank = comm->rank/comm->symrState.lsaSize;
  ans.stride = comm->symrState.lsaSize;
  return ans;
}

NCCL_API_CXX(int, ncclTeamRankToWorld, struct ncclComm* comm, ncclTeam team, int rank);
int ncclTeamRankToWorld(struct ncclComm* comm, ncclTeam team, int rank) {
  return comm->rank + (rank - team.rank)*team.stride;
}

NCCL_API_CXX(int, ncclTeamRankToLsa, struct ncclComm* comm, ncclTeam team, int rank);
int ncclTeamRankToLsa(struct ncclComm* comm, ncclTeam team, int rank) {
  return comm->symrState.lsaSelf + (rank - team.rank)*team.stride;
}
