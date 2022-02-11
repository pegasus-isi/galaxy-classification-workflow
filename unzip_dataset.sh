#!/usr/bin/env bash

cat zipped_data/full_galaxy_data.tar.bz2.part.* > zipped_data/full_galaxy_data.tar.bz2
tar -xjvf zipped_data/full_galaxy_data.tar.bz2
rm -f zipped_data/full_galaxy_data.tar.bz2
