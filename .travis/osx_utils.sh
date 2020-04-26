#!/bin/bash
# See: https://github.com/matthew-brett/multibuild/blob/devel/osx_utils.sh
# See: https://www.python.org/downloads/mac-osx/
LATEST_2p7=2.7.18
LATEST_3p6=3.6.8
LATEST_3p7=3.7.7
LATEST_3p8=3.8.2


function check_var {
    if [ -z "$1" ]; then
        echo "required variable not defined"
        exit 1
    fi
}


function fill_pyver {
    # Convert major or major.minor format to major.minor.micro
    #
    # Hence:
    # 2 -> 2.7.11  (depending on LATEST_2p7 value)
    # 2.7 -> 2.7.11  (depending on LATEST_2p7 value)
    local ver=$1
    check_var $ver
    if [[ $ver =~ [0-9]+\.[0-9]+\.[0-9]+ ]]; then
        # Major.minor.micro format already
        echo $ver
    elif [ $ver == 2 ] || [ $ver == "2.7" ]; then
        echo $LATEST_2p7
    elif [ $ver == 3 ] || [ $ver == "3.7" ]; then
        echo $LATEST_3p7
    elif [ $ver == "3.6" ]; then
        echo $LATEST_3p6
    elif [ $ver == "3.8" ]; then
        echo $LATEST_3p8
    else
        echo "Can't fill version $ver" 1>&2
        exit 1
    fi
}

