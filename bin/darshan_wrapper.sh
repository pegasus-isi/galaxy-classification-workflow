#!/usr/bin/env bash

export DARSHAN_JOBID=CONDOR_JOBID
export DARSHAN_ENABLE_NONMPI=1
env LD_PRELOAD=/nfs/shared/panorama/darshan/lib/libdarshan.so ./$@

