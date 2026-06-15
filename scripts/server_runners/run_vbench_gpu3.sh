#!/bin/bash
bash /data/disk2/timur/run_vbench_dim.sh subject_consistency /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 3
bash /data/disk2/timur/run_vbench_dim.sh background_consistency /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 3
bash /data/disk2/timur/run_vbench_dim.sh subject_consistency /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 3
bash /data/disk2/timur/run_vbench_dim.sh background_consistency /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 3
echo GPU3_DONE
