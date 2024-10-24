# run generate_tasks.py to generate tasks/tasks.csv

python3 generate_tasks.py

# count the number of lines in tasks.csv and store it in a variable
num_tasks=$(wc -l < tasks.csv)

# loop through the number of tasks
for i in $(seq 2 $num_tasks)
do
    # submit each task to the cluster
    qsub sub_one.sh $i
done