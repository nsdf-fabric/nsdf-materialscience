# Expand the template into multiple files, one for each item to be processed

#!/bin/sh

#For openshift change from ks-jobs to jobs and ks-multi-node-job.yaml to multi-node-job.yaml

mkdir ./ks-jobs
for i in 112536 112538 112539 112541
do
  name=radiographic_scan_id_$i
  echo $name
  cat multi_node_job.yaml | sed "s/\$FILE/$name/g" |  sed "s/\$JOB/$i/g" > ./ks-jobs/ms-$name.yaml
done
