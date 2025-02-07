#!/bin/bash

set +e  # Disable exit on error
source test/scripts/ci/ci-utils.sh

: ${JUNIT:="junit_results.xml"}
if ! has_testsuite_closing_tag "$JUNIT"; then
    label=$(tail -n 1 $JUNIT | grep -oP 'name="\K[^"]*')
    start=$(tail -n 1 $JUNIT | grep -oP 'start_time="\K[^"]*')
    move_repro_script $label
    end=$(date +%s%N)
    let runtime=$((end - start))/1000000000
    echo "Error: The XML file is incomplete. The </testsuite> tag is missing from $label. Appending failure to the last testcase."
    echo " status=\"failed\" time=\"$runtime\"><failed>Script timed out</failed></testcase>" >> $xml_file
    end_junit_file
else
    echo "The XML file is complete."
fi