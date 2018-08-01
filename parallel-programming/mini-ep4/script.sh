dataset_bakery="bakery.csv"
dataset_gate="gate.csv"
echo "num_threads,total_time,running_time,avg_accesses,stddev_accesses" > $dataset_bakery
echo "num_threads,total_time,running_time,avg_accesses,stddev_accesses" > $dataset_gate

# ./main nthreads cs_time algorithm
for num_threads in $(seq 10 10 100); do
	for total_time in $(seq 3000000 1000000 10000000); do
	    ./main $num_threads $total_time 0 >> $dataset_bakery
	    ./main $num_threads $total_time 1 >> $dataset_gate
	done
done