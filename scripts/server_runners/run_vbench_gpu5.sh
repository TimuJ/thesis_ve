#!/bin/bash
bash /data/disk2/timur/run_vbench_dim.sh human_action /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 5
bash /data/disk2/timur/run_vbench_dim.sh spatial_relationship /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 5
bash /data/disk2/timur/run_vbench_dim.sh scene /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 5
bash /data/disk2/timur/run_vbench_dim.sh human_action /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 5
bash /data/disk2/timur/run_vbench_dim.sh spatial_relationship /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 5
bash /data/disk2/timur/run_vbench_dim.sh scene /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 5
echo GPU5_DONE
