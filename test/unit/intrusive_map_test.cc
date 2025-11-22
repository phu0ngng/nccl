/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "utils.h"
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <vector>

// Test result tracking
struct TestResults {
  int passed;
  int failed;
  int total;
};

void printTestResult(TestResults* results, const char* testName, bool passed) {
  results->total++;
  if (passed) {
    results->passed++;
    printf("  [PASS] %s\n", testName);
  } else {
    results->failed++;
    printf("  [FAIL] %s\n", testName);
  }
}

// Test object for intrusive map
struct TestObject {
  void* key;
  int value;
  struct TestObject* next;
};

// Test 1: Basic insert and find
bool testBasicIntrusiveMap(TestResults* results) {
  printf("\n=== Basic Insert and Find ===\n");

  ncclIntruAddressMap<TestObject, void*, &TestObject::key, &TestObject::next> map = {};
  bool allPassed = true;

  // Create test objects
  TestObject obj1 = {(void*)0x1000, 42, nullptr};
  TestObject obj2 = {(void*)0x2000, 84, nullptr};
  TestObject obj3 = {(void*)0x3000, 126, nullptr};

  // Insert objects
  ncclResult_t ret = ncclIntruAddressMapInsert(&map, (void*)0x1000, &obj1);
  allPassed &= (ret == ncclSuccess);
  printTestResult(results, "Insert obj1", ret == ncclSuccess);

  ret = ncclIntruAddressMapInsert(&map, (void*)0x2000, &obj2);
  allPassed &= (ret == ncclSuccess);
  printTestResult(results, "Insert obj2", ret == ncclSuccess);

  ret = ncclIntruAddressMapInsert(&map, (void*)0x3000, &obj3);
  allPassed &= (ret == ncclSuccess);
  printTestResult(results, "Insert obj3", ret == ncclSuccess);

  // Find objects
  TestObject* found = nullptr;
  ret = ncclIntruAddressMapFind(&map, (void*)0x1000, &found);
  allPassed &= (ret == ncclSuccess && found == &obj1 && found->value == 42);
  printTestResult(results, "Find obj1", ret == ncclSuccess && found == &obj1);

  found = nullptr;
  ret = ncclIntruAddressMapFind(&map, (void*)0x2000, &found);
  allPassed &= (ret == ncclSuccess && found == &obj2 && found->value == 84);
  printTestResult(results, "Find obj2", ret == ncclSuccess && found == &obj2);

  found = nullptr;
  ret = ncclIntruAddressMapFind(&map, (void*)0x3000, &found);
  allPassed &= (ret == ncclSuccess && found == &obj3 && found->value == 126);
  printTestResult(results, "Find obj3", ret == ncclSuccess && found == &obj3);

  // Verify count
  allPassed &= (map.base.count == 3);
  printTestResult(results, "Correct count after inserts", map.base.count == 3);

  // Clean up
  ncclIntruAddressMapRemove(&map, (void*)0x1000);
  ncclIntruAddressMapRemove(&map, (void*)0x2000);
  ncclIntruAddressMapRemove(&map, (void*)0x3000);
  ncclIntruAddressMapDestruct(&map);

  return allPassed;
}

// Test 2: Remove operations
bool testRemove(TestResults* results) {
  printf("\n=== Remove Operations ===\n");

  ncclIntruAddressMap<TestObject, void*, &TestObject::key, &TestObject::next> map = {};
  bool allPassed = true;

  // Create and insert several objects
  TestObject objects[5];
  void* keys[5] = {(void*)0x1000, (void*)0x2000, (void*)0x3000, (void*)0x4000, (void*)0x5000};

  for (int i = 0; i < 5; i++) {
    objects[i].key = keys[i];
    objects[i].value = i + 100;
    objects[i].next = nullptr;
    ncclIntruAddressMapInsert(&map, keys[i], &objects[i]);
  }

  // Remove middle entry
  ncclResult_t ret = ncclIntruAddressMapRemove(&map, keys[2]);
  allPassed &= (ret == ncclSuccess);
  printTestResult(results, "Remove middle entry", ret == ncclSuccess);

  allPassed &= (map.base.count == 4);
  printTestResult(results, "Correct count after remove", map.base.count == 4);

  // Verify removed entry is not found
  TestObject* found = nullptr;
  ret = ncclIntruAddressMapFind(&map, keys[2], &found);
  allPassed &= (ret == ncclSuccess && found == nullptr);
  printTestResult(results, "Removed entry not found", ret == ncclSuccess && found == nullptr);

  // Verify other entries still exist
  found = nullptr;
  ret = ncclIntruAddressMapFind(&map, keys[0], &found);
  allPassed &= (ret == ncclSuccess && found == &objects[0]);
  printTestResult(results, "Other entries still found", ret == ncclSuccess && found == &objects[0]);

  // Remove all remaining entries
  for (int i = 0; i < 5; i++) {
    if (i != 2) {
      ncclIntruAddressMapRemove(&map, keys[i]);
    }
  }

  // Verify table is cleaned up when empty
  allPassed &= (map.base.count == 0 && map.base.hbits == 0 && map.base.table == nullptr);
  printTestResult(results, "Table cleaned up when empty", map.base.count == 0 && map.base.table == nullptr);

  return allPassed;
}

// Test 3: Growth and rehashing
bool testRehashing(TestResults* results) {
  printf("\n=== Growth and Rehashing ===\n");

  ncclIntruAddressMap<TestObject, void*, &TestObject::key, &TestObject::next> map = {};
  bool allPassed = true;

  const int numObjects = 100;
  TestObject* objects = new TestObject[numObjects];

  int initialHbits = 0;

  // Insert many objects
  for (int i = 0; i < numObjects; i++) {
    objects[i].key = (void*)(uintptr_t)(0x1000 + i * 0x100);
    objects[i].value = i;
    objects[i].next = nullptr;

    ncclResult_t ret = ncclIntruAddressMapInsert(&map, objects[i].key, &objects[i]);
    if (ret != ncclSuccess) {
      allPassed = false;
      break;
    }
    if (i == 0) {
      initialHbits = map.base.hbits;
    }
  }

  printTestResult(results, "Insert 100 objects", allPassed && map.base.count == numObjects);

  // Verify table grew
  allPassed &= (map.base.hbits > initialHbits);
  printTestResult(results, "Table grew (hbits increased)", map.base.hbits > initialHbits);

  // Verify all objects can be found after rehashing
  bool allFound = true;
  for (int i = 0; i < numObjects; i++) {
    TestObject* found = nullptr;
    ncclResult_t ret = ncclIntruAddressMapFind(&map, objects[i].key, &found);
    if (ret != ncclSuccess || found != &objects[i]) {
      allFound = false;
      break;
    }
  }

  allPassed &= allFound;
  printTestResult(results, "All objects found after rehashing", allFound);

  // Remove all objects
  for (int i = 0; i < numObjects; i++) {
    ncclIntruAddressMapRemove(&map, objects[i].key);
  }

  ncclIntruAddressMapDestruct(&map);
  delete[] objects;

  return allPassed;
}

// Test 4: Edge cases
bool testEdgeCases(TestResults* results) {
  printf("\n=== Edge Cases ===\n");

  bool allPassed = true;

  // Test 1: Find in empty table
  ncclIntruAddressMap<TestObject, void*, &TestObject::key, &TestObject::next> map1 = {};
  TestObject* found = nullptr;
  ncclResult_t ret = ncclIntruAddressMapFind(&map1, (void*)0x1000, &found);
  allPassed &= (ret == ncclSuccess && found == nullptr);
  printTestResult(results, "Find in empty table returns nullptr", ret == ncclSuccess && found == nullptr);

  // Test 2: Remove from empty table (idempotent)
  ret = ncclIntruAddressMapRemove(&map1, (void*)0x1000);
  allPassed &= (ret == ncclSuccess);
  printTestResult(results, "Remove from empty table succeeds", ret == ncclSuccess);

  // Test 3: Remove non-existent key
  TestObject obj1 = {(void*)0x1000, 100, nullptr};
  ncclIntruAddressMapInsert(&map1, (void*)0x1000, &obj1);
  ret = ncclIntruAddressMapRemove(&map1, (void*)0x9999);
  allPassed &= (ret == ncclSuccess);
  printTestResult(results, "Remove non-existent key succeeds", ret == ncclSuccess);

  // Test 4: Find non-existent key
  found = nullptr;
  ret = ncclIntruAddressMapFind(&map1, (void*)0x9999, &found);
  allPassed &= (ret == ncclSuccess && found == nullptr);
  printTestResult(results, "Find non-existent key returns nullptr", ret == ncclSuccess && found == nullptr);

  // Test 5: Null key (key=0 is valid)
  TestObject obj2 = {(void*)0, 200, nullptr};
  ret = ncclIntruAddressMapInsert(&map1, (void*)0, &obj2);
  allPassed &= (ret == ncclSuccess);
  printTestResult(results, "Insert with key=0", ret == ncclSuccess);

  found = nullptr;
  ret = ncclIntruAddressMapFind(&map1, (void*)0, &found);
  allPassed &= (ret == ncclSuccess && found == &obj2);
  printTestResult(results, "Find with key=0", ret == ncclSuccess && found == &obj2);

  ncclIntruAddressMapRemove(&map1, (void*)0x1000);
  ncclIntruAddressMapRemove(&map1, (void*)0);
  ncclIntruAddressMapDestruct(&map1);

  return allPassed;
}

// Test 5: Collision handling
bool testCollisionHandling(TestResults* results) {
  printf("\n=== Collision Handling ===\n");

  ncclIntruAddressMap<TestObject, void*, &TestObject::key, &TestObject::next> map = {};
  bool allPassed = true;

  // Insert entries that will likely collide in small table
  TestObject objects[8];
  for (int i = 0; i < 8; i++) {
    objects[i].key = (void*)(uintptr_t)(i + 1);
    objects[i].value = i + 1000;
    objects[i].next = nullptr;

    ncclResult_t ret = ncclIntruAddressMapInsert(&map, objects[i].key, &objects[i]);
    if (ret != ncclSuccess) {
      allPassed = false;
    }
  }

  printTestResult(results, "Insert entries with potential collisions", allPassed);

  // Verify all entries can be found
  bool allFound = true;
  for (int i = 0; i < 8; i++) {
    TestObject* found = nullptr;
    ncclResult_t ret = ncclIntruAddressMapFind(&map, objects[i].key, &found);
    if (ret != ncclSuccess || found != &objects[i]) {
      allFound = false;
      break;
    }
  }

  allPassed &= allFound;
  printTestResult(results, "All colliding entries found", allFound);

  // Clean up
  for (int i = 0; i < 8; i++) {
    ncclIntruAddressMapRemove(&map, objects[i].key);
  }
  ncclIntruAddressMapDestruct(&map);

  return allPassed;
}

// Test 6: Sequential stress test
bool testStressSequential(TestResults* results) {
  printf("\n=== Stress Test - Sequential Operations ===\n");

  ncclIntruAddressMap<TestObject, void*, &TestObject::key, &TestObject::next> map = {};
  bool allPassed = true;

  const int numEntries = 10000;
  TestObject* objects = new TestObject[numEntries];

  printf("  Inserting %d entries...\n", numEntries);

  // Insert many entries sequentially
  for (int i = 0; i < numEntries; i++) {
    objects[i].key = (void*)(uintptr_t)(i * 4096);  // Page-aligned addresses
    objects[i].value = i;
    objects[i].next = nullptr;

    ncclResult_t ret = ncclIntruAddressMapInsert(&map, objects[i].key, &objects[i]);
    if (ret != ncclSuccess) {
      printf("  Failed to insert entry %d\n", i);
      allPassed = false;
      break;
    }
  }

  printTestResult(results, "Insert 10000 entries", allPassed && map.base.count == numEntries);
  printf("  Final table: %d entries, %d hbits (size=%d)\n",
         map.base.count, map.base.hbits, 1 << map.base.hbits);

  // Verify all entries
  printf("  Verifying all entries...\n");
  bool allFound = true;
  for (int i = 0; i < numEntries; i++) {
    TestObject* found = nullptr;
    ncclResult_t ret = ncclIntruAddressMapFind(&map, objects[i].key, &found);
    if (ret != ncclSuccess || found != &objects[i]) {
      allFound = false;
      break;
    }
  }

  allPassed &= allFound;
  printTestResult(results, "Find all 10000 entries", allFound);

  // Remove all entries
  for (int i = 0; i < numEntries; i++) {
    ncclIntruAddressMapRemove(&map, objects[i].key);
  }

  ncclIntruAddressMapDestruct(&map);
  delete[] objects;

  return allPassed;
}

// Test 7: Random operations stress test
bool testStressRandom(TestResults* results) {
  printf("\n=== Stress Test - Random Operations ===\n");

  ncclIntruAddressMap<TestObject, void*, &TestObject::key, &TestObject::next> map = {};
  bool allPassed = true;

  const int maxObjects = 5000;
  TestObject* pool = new TestObject[maxObjects];
  std::vector<int> activeIndices;

  srand(time(nullptr));

  int insertCount = 0;
  int removeCount = 0;
  int findCount = 0;

  for (int op = 0; op < 20000; op++) {
    int choice = rand() % 3;

    if (choice == 0 || activeIndices.empty()) {
      // Insert
      if (insertCount < maxObjects) {
        int idx = insertCount;
        pool[idx].key = (void*)(uintptr_t)(rand() * 1000LL + rand());
        pool[idx].value = idx;
        pool[idx].next = nullptr;

        ncclResult_t ret = ncclIntruAddressMapInsert(&map, pool[idx].key, &pool[idx]);
        if (ret == ncclSuccess) {
          activeIndices.push_back(idx);
          insertCount++;
        } else {
          allPassed = false;
        }
      }
    } else if (choice == 1 && !activeIndices.empty()) {
      // Remove
      int vecIdx = rand() % activeIndices.size();
      int idx = activeIndices[vecIdx];

      ncclResult_t ret = ncclIntruAddressMapRemove(&map, pool[idx].key);
      if (ret == ncclSuccess) {
        activeIndices.erase(activeIndices.begin() + vecIdx);
        removeCount++;
      } else {
        allPassed = false;
      }
    } else if (!activeIndices.empty()) {
      // Find
      int vecIdx = rand() % activeIndices.size();
      int idx = activeIndices[vecIdx];
      TestObject* found = nullptr;

      ncclResult_t ret = ncclIntruAddressMapFind(&map, pool[idx].key, &found);
      if (ret == ncclSuccess && found == &pool[idx]) {
        findCount++;
      } else {
        allPassed = false;
      }
    }
  }

  printf("  Operations: %d inserts, %d removes, %d finds\n", insertCount, removeCount, findCount);
  printf("  Final table: %d entries, %d hbits\n", map.base.count, map.base.hbits);

  allPassed &= (map.base.count == (int)activeIndices.size());
  printTestResult(results, "Correct count after random operations", map.base.count == (int)activeIndices.size());

  // Verify all remaining objects can be found
  bool allFound = true;
  for (size_t i = 0; i < activeIndices.size(); i++) {
    TestObject* found = nullptr;
    int idx = activeIndices[i];
    ncclResult_t ret = ncclIntruAddressMapFind(&map, pool[idx].key, &found);
    if (ret != ncclSuccess || found != &pool[idx]) {
      allFound = false;
      break;
    }
  }

  allPassed &= allFound;
  printTestResult(results, "All remaining objects found", allFound);

  // Clean up
  for (size_t i = 0; i < activeIndices.size(); i++) {
    ncclIntruAddressMapRemove(&map, pool[activeIndices[i]].key);
  }

  ncclIntruAddressMapDestruct(&map);
  delete[] pool;

  return allPassed;
}

// Test 8: Performance benchmark
bool testPerformance(TestResults* results) {
  printf("\n=== Performance Benchmark ===\n");

  ncclIntruAddressMap<TestObject, void*, &TestObject::key, &TestObject::next> map = {};

  const int numEntries = 100000;
  TestObject* objects = new TestObject[numEntries];

  // Benchmark insertions
  clock_t start = clock();
  for (int i = 0; i < numEntries; i++) {
    objects[i].key = (void*)(uintptr_t)(i * 8);
    objects[i].value = i;
    objects[i].next = nullptr;
    ncclIntruAddressMapInsert(&map, objects[i].key, &objects[i]);
  }
  clock_t end = clock();
  double insertTime = ((double)(end - start)) / CLOCKS_PER_SEC;

  // Benchmark lookups
  start = clock();
  for (int i = 0; i < numEntries; i++) {
    TestObject* found = nullptr;
    ncclIntruAddressMapFind(&map, objects[i].key, &found);
  }
  end = clock();
  double findTime = ((double)(end - start)) / CLOCKS_PER_SEC;

  // Benchmark removes
  start = clock();
  for (int i = 0; i < numEntries; i++) {
    ncclIntruAddressMapRemove(&map, objects[i].key);
  }
  end = clock();
  double removeTime = ((double)(end - start)) / CLOCKS_PER_SEC;

  printf("  Insert  %d entries: %.3f seconds (%.0f ops/sec)\n",
         numEntries, insertTime, numEntries / insertTime);
  printf("  Find    %d entries: %.3f seconds (%.0f ops/sec)\n",
         numEntries, findTime, numEntries / findTime);
  printf("  Remove  %d entries: %.3f seconds (%.0f ops/sec)\n",
         numEntries, removeTime, numEntries / removeTime);

  printTestResult(results, "Performance benchmark completed", true);

  delete[] objects;
  return true;
}

int main(int argc, char* argv[]) {
  printf("\n");
  printf("╔════════════════════════════════════════════════════════════════╗\n");
  printf("║      NCCL Intrusive Address Map Unit Test Suite               ║\n");
  printf("╚════════════════════════════════════════════════════════════════╝\n");

  TestResults results = {0, 0, 0};

  // Run all tests
  testBasicIntrusiveMap(&results);
  testRemove(&results);
  testRehashing(&results);
  testEdgeCases(&results);
  testCollisionHandling(&results);
  testStressSequential(&results);
  testStressRandom(&results);
  testPerformance(&results);

  // Print summary
  printf("\n");
  printf("╔════════════════════════════════════════════════════════════════╗\n");
  printf("║                      Test Summary                              ║\n");
  printf("╠════════════════════════════════════════════════════════════════╣\n");
  printf("║  Total Tests:   %3d                                            ║\n", results.total);
  printf("║  Passed:        %3d                                            ║\n", results.passed);
  printf("║  Failed:        %3d                                            ║\n", results.failed);
  printf("╚════════════════════════════════════════════════════════════════╝\n");
  printf("\n");

  if (results.failed == 0) {
    printf("✓ All tests passed!\n\n");
    return 0;
  } else {
    printf("✗ Some tests failed!\n\n");
    return 1;
  }
}

