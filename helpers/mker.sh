DIR0=aperf
mkdir $DIR0
for w in w1 w7 w5 w0
do
    mkdir $DIR0/$w 
    for kcs in kcs300 kcs121 kcs331 kcs222
    do
        mkdir $DIR0/$w/$kcs
    done
done

echo "Done"
