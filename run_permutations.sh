#!/bin/bash

# 依次执行各个 perm 参数：nohup bash run_permutations.sh > run_permutations_t4.log 2>&1 &
for perm in perm0 perm1 perm2 perm3 perm4 perm5 perm6 perm7
do
    echo "Running with --perm $perm"
    python train_clner.py --gpu 0 --m mlm --corpus conll --setup split --perm $perm --label_map_path data/conll/label_map_timesup_ratio0.6_multitoken_top6.json --return_entity_level_metrics
#    python train_clner.py --gpu 0 --m mlm --corpus conll --setup split --perm $perm --label_map_path data/conll/label_map_timesup_ratio0.6_multitoken_top6.json --return_entity_level_metrics --std filter
    echo "Finished running with --perm $perm"
done

echo "All permutations have been processed."

#for split_seed in 1 2 3 4 5 6
#do
#    echo "Running experiments for split_seed $split_seed"
#
#    for perm in perm0 perm1 perm2 perm3 perm4 perm5 perm6 perm7
#    do
#        echo "Running with --perm $perm and --split_seed $split_seed"
#        python train_clner.py --gpu 0 --m mlm --corpus conll --setup split --perm $perm --split_seed $split_seed --label_map_path data/conll/label_map_timesup_ratio0.6_multitoken_top6.json --return_entity_level_metrics
#        echo "Finished running with --perm $perm and --split_seed $split_seed"
#    done
#
#    echo "Finished all permutations for split_seed $split_seed"
#done
#
#echo "All split seeds and permutations have been processed."