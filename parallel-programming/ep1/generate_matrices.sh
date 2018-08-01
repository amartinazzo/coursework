m=11
p=13
n=7

matrixD="D.txt"
matrixE="E.txt"
matrixDD="DD.txt"
matrixEE="EE.txt"
echo "$m $p" > $matrixD
echo "$p $n" > $matrixE
echo "" > $matrixDD
echo "" > $matrixEE
 
for x in $(seq 1 1 $m); do
	for y in $(seq 1 1 $p); do
	    echo "$x $y $x" >> $matrixD
	    # echo "$num \c" >> $matrixDD
	done
	done

for x in $(seq 1 1 $p); do
	for y in $(seq 1 1 $n); do
	    echo "$x $y $x" >> $matrixE
	    # echo "$num \c" >> $matrixEE
	done
done