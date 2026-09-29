#!/bin/bash
header="/* -------------------------------------------------------------------------------
 * Tomocam Copyright (c) 2018
 *
 * The Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals from the
 * U.S. Dept. of Energy). All rights reserved.
 *
 * If you have questions about your rights to use or distribute this software,
 * please contact Berkeley Lab's Innovation & Partnerships Office at
 * IPO@lbl.gov.
 *
 * NOTICE. This Software was developed under funding from the U.S. Department of
 * Energy and the U.S. Government consequently retains certain rights. As such,
 * the U.S. Government has been granted for itself and others acting on its
 * behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
 * to reproduce, distribute copies to the public, prepare derivative works, and
 * perform publicly and display publicly, and to permit other to do so.
 *---------------------------------------------------------------------------------
 */
"
copyright='Tomocam Copyright (c) 2018'
files=$(find ./src -type f)

# find .cpp, .h, .cu, and .cuh files in src directory
files=$(find ./src -type f -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh")
for file in $files; do
    if ! grep -q "$copyright" $file; then
        echo "Adding license to file $file"
        echo "${header}" > $file.tmp
        cat $file >> $file.tmp && mv $file.tmp $file
    fi
done
