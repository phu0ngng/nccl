#include <stdio.h>
#include "../../build/include/nccl_net.h"

extern ncclNet_t NCCL_PLUGIN_SYMBOL;
extern ncclCollNet_t NCCL_COLLNET_PLUGIN_SYMBOL;

int main() {
  printf("Net plugin : %s\n", NCCL_PLUGIN_SYMBOL.name);
  printf("Coll net plugin : %s\n", NCCL_COLLNET_PLUGIN_SYMBOL.name);
  return 0;
}
