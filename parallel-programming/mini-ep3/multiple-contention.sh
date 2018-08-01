for vector_size in $(seq 1 50 1000); do
	echo "sampling vector of size $vector_size..."
	for n_threads in $(seq 1 50 1000); do
	    bash ./contention.sh $vector_size $n_threads
	done
done

# for n in $(seq 1 500); do
# 	bash ./contention.sh 200 10
# done

# echo "frequencies of min times"
# sort -n dataset-min2.csv | uniq -c
# echo "frequencies of max times"
# sort -n dataset-max2.csv | uniq -c