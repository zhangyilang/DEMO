file=$1
nohup ~/MATLAB_R2016a/bin/matlab -nodisplay -r ${file%.m} > log.out < /dev/null &
