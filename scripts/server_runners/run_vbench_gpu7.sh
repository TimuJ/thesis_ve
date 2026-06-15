#!/bin/bash
bash /data/disk2/timur/run_vbench_dim.sh appearance_style /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 7
bash /data/disk2/timur/run_vbench_dim.sh temporal_style /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 7
bash /data/disk2/timur/run_vbench_dim.sh overall_consistency /data/disk2/timur/results/vbench2_all_input /data/disk2/timur/results/vbench2_mgld_all 7
bash /data/disk2/timur/run_vbench_dim.sh aesthetic_quality /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 7
bash /data/disk2/timur/run_vbench_dim.sh dynamic_degree /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 7
bash /data/disk2/timur/run_vbench_dim.sh appearance_style /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 7
bash /data/disk2/timur/run_vbench_dim.sh temporal_style /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 7
bash /data/disk2/timur/run_vbench_dim.sh overall_consistency /data/disk2/timur/results/vbench2_lq_input /data/disk2/timur/results/vbench2_lq_all 7
echo GPU7_DONE
