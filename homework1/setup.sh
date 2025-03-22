#!/bin/bash

main() {
    local -r cmddir=$(dirname $(readlink -f ${0}))
    pushd ${cmddir:-~}
    curl -L "https://drive.switch.ch/index.php/s/zv4Zuv0AFZJx42B/download" -o homework1.zip
    unzip -o homework1.zip
    rm -rf homework1.zip
    popd
}

main "$@"
