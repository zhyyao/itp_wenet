#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.

root=wenet
example=aishell/s0
src_dir=data
trg_dir=/datablob/am_data/E2E/v-zhuoyyao/aishell/source

stage=0 # start from 0 if you need to start from data preparation
stop_stage=3

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0
nj=16

dict=$trg_dir/lang_char.txt

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

data_sets="train dev test"
train_set=train
. $root/examples/$example/path.sh || exit 1;
cd $root/examples/$example
train_config=conf/train_conformer.yaml

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  mkdir -p $trg_dir
  # remove the space between the text labels for Mandarin dataset
  for x in ${data_sets}; do
    mkdir $trg_dir/${x}
    cp $src_dir/${x}/wav.scp $trg_dir/${x}
    cp $src_dir/${x}/text $trg_dir/${x}/text.org
    paste -d " " <(cut -f 1 -d" " $trg_dir/${x}/text.org) \
      <(cut -f 2- -d" " $trg_dir/${x}/text.org | tr -d " ") \
      > $trg_dir/${x}/text
    rm $trg_dir/${x}/text.org
  done

  tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp $trg_dir/${train_set}/wav.scp \
    --out_cmvn $trg_dir/$train_set/global_cmvn
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<blank> 0" > ${dict}  # 0 is for "blank" in CTC
  echo "<unk> 1"  >> ${dict}  # <unk> must be 1
  tools/text2token.py -s 1 -n 1 $trg_dir/${train_set}/text | cut -f 2- -d" " \
    | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | \
    awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Prepare data, prepare required format"
  for x in ${data_sets}; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 ${trg_dir}/$x/wav.scp ${trg_dir}/$x/text \
        $(realpath $trg_dir/$x/shards) $trg_dir/$x/data.list
    else
      tools/make_raw_list.py $trg_dir/$x/wav.scp $trg_dir/$x/text \
        $trg_dir/$x/data.list
    fi
  done
fi


