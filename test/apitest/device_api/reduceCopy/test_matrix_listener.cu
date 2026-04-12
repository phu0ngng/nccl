/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Dedicated translation unit for test matrix verification and listener.
#include "test_reduce_copy.h"
#include <cstdlib>

// Custom test event listener to run matrix verification after all tests complete
class MatrixTestListener : public ::testing::EmptyTestEventListener {
private:
  bool matrixVerificationFailed_ = false;

public:
  void OnTestProgramEnd(const ::testing::UnitTest& unit_test) override {
#if !defined(__CUDA_ARCH__)
    // Skip entirely if no reduceCopy types were selected at runtime.
    // Empty NCCL_TEST_REDUCE_COPY_TYPES means no types were compiled in;
    // printing a matrix with 0 executed / N expected and exiting with failure
    // would break regular device API test runs that don't run reduceCopy tests.
    const char* envTypes = std::getenv("NCCL_TEST_REDUCE_COPY_TYPES");
    if (!envTypes || std::string(envTypes).empty()) return;

    // Run matrix verification after all tests have completed
    TestMatrix& matrix = TestMatrix::getInstance();
    TestMatrix::Stats stats = matrix.getStats();
    std::vector<TestMatrix::TestKey> missing = matrix.getMissingTests();
    std::vector<std::string> unknownTypes;
    std::vector<std::string> notSelectedTypes = TestMatrix::getNotSelectedTypeNames(&unknownTypes);
    std::vector<TestMatrix::TypeCoverage> typeCoverage = matrix.getTypeCoverage();

    // Print short summary (always)
    printf("\n");
    printf("========================================\n");
    printf("TEST COVERAGE MATRIX SUMMARY\n");
    printf("========================================\n");
    printf("Total Expected Tests: %zu\n", stats.totalExpected);
    printf("Total Executed Tests: %zu\n", stats.totalExecuted);
    printf("Total Missing Tests:  %zu\n", stats.totalMissing);
    printf("Coverage: %.1f%%\n",
         stats.totalExpected > 0 ?
         (100.0 * stats.totalExecuted / stats.totalExpected) : 0.0);
    printf("========================================\n");

    printf("Type Selection (NCCL_TEST_REDUCE_COPY_TYPES): %s\n", envTypes);
    if (!unknownTypes.empty()) {
      printf("Unknown type tokens (ignored): ");
      for (size_t i = 0; i < unknownTypes.size(); ++i) {
        printf("%s%s", unknownTypes[i].c_str(), (i + 1 < unknownTypes.size()) ? ", " : "");
      }
      printf("\n");
    }
    if (!notSelectedTypes.empty()) {
      printf("Types not tested: ");
      for (size_t i = 0; i < notSelectedTypes.size(); ++i) {
        printf("%s%s", notSelectedTypes[i].c_str(), (i + 1 < notSelectedTypes.size()) ? ", " : "");
      }
      printf("\n");
    }

    if (!typeCoverage.empty()) {
      printf("\nTYPE COVERAGE:\n");
      printf("----------------------------------------\n");
      for (const auto& entry : typeCoverage) {
        printf("  %s: %zu/%zu (missing %zu) = %.1f%%\n",
             entry.typeName.c_str(),
             entry.executed, entry.expected, entry.missing, entry.coverage);
      }
      printf("----------------------------------------\n");
    }

    // Print detailed missing tests list only when NCCL_TEST_REDUCE_COPY_LIST_MISSING=1
    if (!missing.empty()) {
      if (isMissingTestsListEnabled()) {
        printf("\nMISSING TEST COMBINATIONS:\n");
        printf("----------------------------------------\n");
        for (const auto& key : missing) {
          printf("  - %s\n", TestMatrix::keyToString(key).c_str());
        }
        printf("----------------------------------------\n");
      } else {
        printf("\nNote: Run with NCCL_TEST_REDUCE_COPY_LIST_MISSING=1 to see detailed list of %zu missing test(s).\n",
             stats.totalMissing);
      }
    }

    printf("\n");

    // Store verification result and fail the run if tests are missing
    matrixVerificationFailed_ = (stats.totalExpected != stats.totalExecuted || !missing.empty());

    if (matrixVerificationFailed_) {
      printf("ERROR: Not all expected test combinations were executed. "
           "Missing %zu test(s).\n", stats.totalMissing);
      // Exit with failure so the test run reports an error (listener runs after all
      // tests, so no test can fail here; exiting is the only way to report failure).
      // Cleanup is already done: OnTestProgramEnd runs after all tests and after
      // each suite's TearDownTestCase(), so NCCL comms and CUDA streams are released.
      exit(EXIT_FAILURE);
    }
#endif // !__CUDA_ARCH__
  }

  bool hasVerificationFailed() const {
    return matrixVerificationFailed_;
  }
};

// Global pointer to access the listener
static MatrixTestListener* g_matrix_listener = nullptr;

// This test may run at any time during test execution, so it checks if all tests
// have completed. If not all tests have run yet, it skips the verification.
// The actual matrix summary and final verification is done by MatrixTestListener::OnTestProgramEnd()
// which is guaranteed to run after all tests complete.
TEST(ReduceCopyTestCoverage, VerifyTestMatrix) {
#if !defined(__CUDA_ARCH__)
  {
    const char* envTypes = std::getenv("NCCL_TEST_REDUCE_COPY_TYPES");
    if (!envTypes || std::string(envTypes).empty()) return;
  }
  TestMatrix& matrix = TestMatrix::getInstance();
  TestMatrix::Stats stats = matrix.getStats();
  std::vector<TestMatrix::TestKey> missing = matrix.getMissingTests();

  // Check if all expected tests have been executed
  // If not, this test is running too early - wait for all tests to complete
  // The listener will do the final verification after all tests complete
  if (stats.totalExecuted < stats.totalExpected) {
    // Not all tests have run yet - this is expected if Google Test runs this test early
    // Don't fail yet - the listener will verify at the end and print the summary
    // Just record that we checked early
    printf("[TEST] Note: VerifyTestMatrix ran early (%zu/%zu tests executed). "
         "Final verification will be done after all tests complete.\n",
         stats.totalExecuted, stats.totalExpected);
    // Return early - don't do assertions yet
    return;
  }

  // If we reach here, all tests should have completed
  // Verify all expected tests were executed
  EXPECT_EQ(stats.totalExpected, stats.totalExecuted)
    << "Not all expected test combinations were executed. "
    << "Missing " << stats.totalMissing << " test(s). "
    << "See test output above for detailed matrix summary.";

  // Also verify no missing tests
  EXPECT_TRUE(missing.empty())
    << "Missing " << missing.size() << " test combination(s). "
    << "See test output above for details.";

  // Also check if the listener detected any failures
  if (g_matrix_listener && g_matrix_listener->hasVerificationFailed()) {
    FAIL() << "Test matrix verification failed. See summary above for details.";
  }
#endif // !__CUDA_ARCH__
}

// Register the event listener to run matrix verification after all tests
// This ensures the matrix summary is printed even if the test fails or is skipped
// The listener is registered via a static initializer that runs before main()
namespace {
  struct MatrixListenerRegistrar {
    MatrixListenerRegistrar() {
      g_matrix_listener = new MatrixTestListener();
      ::testing::UnitTest::GetInstance()->listeners().Append(g_matrix_listener);
    }
  };
  // Static initializer ensures listener is registered before any tests run
  static MatrixListenerRegistrar g_matrix_listener_registrar;
}
