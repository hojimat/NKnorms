DIR0=allperf
DIR1=aperf

find . -path './'$DIR0'/P*NSOC4*W[1*.csv'   -exec cp {} $DIR1/w1 \;
find . -path './'$DIR0'/P*NSOC4*W[0.7*.csv' -exec cp {} $DIR1/w7 \;
find . -path './'$DIR0'/P*NSOC4*W[0.5*.csv' -exec cp {} $DIR1/w5 \;
find . -path './'$DIR0'/P*NSOC4*W[0.0*.csv' -exec cp {} $DIR1/w0 \;
echo "Level 1: W"

for w in w1 w7 w5 w0
do
    find . -path './'$DIR1/$w'/P*RHO0.3*.csv' -exec mv {} $DIR1/$w/rho3 \;
    find . -path './'$DIR1/$w'/P*RHO0.9*.csv' -exec mv {} $DIR1/$w/rho9 \;
    
done
echo "Level 2: RHO"

for w in w1 w7 w5 w0
do
    for rho in rho3 rho9
    do
        find . -path './'$DIR1/$w/$rho/'P*K3C0S0*.csv' -exec mv {} $DIR1/$w/$rho/kcs300 \;
        find . -path './'$DIR1/$w/$rho/'P*K1C2S1*.csv' -exec mv {} $DIR1/$w/$rho/kcs121 \;
        find . -path './'$DIR1/$w/$rho/'P*K3C3S1*.csv' -exec mv {} $DIR1/$w/$rho/kcs331 \;
        find . -path './'$DIR1/$w/$rho/'P*K2C2S2*.csv' -exec mv {} $DIR1/$w/$rho/kcs222 \;
    done
done
echo "Level 3: KCS"
