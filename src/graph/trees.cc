/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"

#define RANK_TO_INDEX(r) (rank > root ? rank-1 : rank)

/* Btree which alternates leaves and nodes, while trying to minimize distance between connected ranks.
 *
 * Algorithm:
 * 1. Find our level in the tree
 * 2. Find ranks down. Should be rank +/- 2^(level-1)
 * 3. Find our rank up. Needs a bit more bit manipulations, to clear bit (level+1) and set bit (level).
 * 4. In the case where the tree is not complete, make sure we re-attach correctly ranks .
 *
 * Illustration :
 *                 7
 *          ______/ \______
 *         3               11
 *       /   \           /   \
 *     1       5       9      \
 *    / \     / \     / \      \
 *   0   2   4   6   8  10     12
 */

int getRankUp(int rank, int level) {
  return (rank + (1<<level)) & ~(1<<(level+1));
}

#include <stdio.h>
ncclResult_t ncclGetBtree(int nranks, int rank, int* u, int* d0, int* d1, int* parentChildType) {
  int level = 0;
  for (int bit=1; bit<nranks && (bit&rank); bit<<=1) level++;

  // Find rank down: should be rank +/- 2^level.
  if (level == 0) *d0 = *d1 = -1;
  else {
    int l = level - 1;
    *d0 = rank - (1<<l);
    if (rank == nranks-1) {
      *d1 = -1;
    } else {
      *d1 = rank + (1<<l);
      // If d1 is out of bounds, go down until we are within 0..nranks-1
      l--;
      while (*d1 >= nranks) *d1 = *d1 - (1<<(l--));
    }
  }

  // Find rank up.
  // The most ranks we can have with N levels is 2^(N+1)-1.
  if (nranks < 1<<(level+1)) {
    *u = -1; // I'm the top rank
  } else {
    int up = rank, l = level;
    // If the up rank is out of bounds, continue to go up until we're within 0..nranks-1.
    do up = getRankUp(up, l++); while (up >= nranks);
    *u = up;
    *parentChildType = rank > up ? 1 : 0;
  }
  return ncclSuccess;
}

/* Build a double binary tree. Take the previous tree for the first tree.
 * For the second tree, we use a mirror tree (if nranks is even)
 *
 *                 7                   2
 *          ______/ \                 / \______
 *         3         \               /         6
 *       /   \        \             /        /   \
 *     1       5       9           0        4      8
 *    / \     / \     /             \      / \    / \
 *   0   2   4   6   8               1    3   5  7   9
 *
 * or shift it by one rank (if nranks is odd).
 *
 *                 7                            8
 *          ______/ \                    ______/ \
 *         3         \                  4         \
 *       /   \        \               /   \        \
 *     1       5       9            2       6       10
 *    / \     / \     / \          / \     / \     / \
 *   0   2   4   6   8  10        1   3   5   7   9   0
 */
ncclResult_t ncclGetDtree(int nranks, int rank, int* s0, int* d0_0, int* d0_1, int* parentChildType0, int* s1, int* d1_0, int* d1_1, int* parentChildType1) {
  // First tree ... use a btree
  ncclGetBtree(nranks, rank, s0, d0_0, d0_1, parentChildType0);
  // Second tree ... mirror or shift
  if (nranks % 2 == 1) {
    // shift
    int shiftrank = (rank-1+nranks) % nranks;
    int u, d0, d1;
    ncclGetBtree(nranks, shiftrank, &u, &d0, &d1, parentChildType1);
    *s1 = u == -1 ? -1 : (u+1) % nranks;
    *d1_0 = d0 == -1 ? -1 : (d0+1) % nranks;
    *d1_1 = d1 == -1 ? -1 : (d1+1) % nranks;
  } else {
    // mirror
    int u, d0, d1;
    ncclGetBtree(nranks, nranks-1-rank, &u, &d0, &d1, parentChildType1);
    *s1 = u == -1 ? -1 : nranks-1-u;
    *d1_0 = d0 == -1 ? -1 : nranks-1-d0;
    *d1_1 = d1 == -1 ? -1 : nranks-1-d1;
  }
  return ncclSuccess;
}
