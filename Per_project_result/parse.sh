#$1=data.csv
#$2="RF-clean.csv"
while read line
do
    ndod=$(grep -r "$line,NDOD" $2)
    f_score_0=$(echo $ndod | cut -d',' -f3)
    nod=$(grep -r "$line,NOD" $2)
    f_score_1=$(echo $nod | cut -d',' -f3)
    od=$(grep -r "$line,OD" $2)
    f_score_2=$(echo $od | cut -d',' -f3)
    nio=$(grep -r "$line,NIO" $2)
    f_score_3=$(echo $nio | cut -d',' -f3)
    id=$(grep -r "$line,ID" $2)
    f_score_4=$(echo $id | cut -d',' -f3)
    ud=$(grep -r "$line,UD" $2)
    f_score_5=$(echo $ud | cut -d',' -f3)
    echo $f_score_0,$f_score_1,$f_score_2,$f_score_3,$f_score_4,$f_score_5 >> B1
done < "data.csv"
