from pyhive import hive
host_name = "192.168.1.10"
port = 10000
user = "cloudera"
password = "cloudera"
database="default"
import numpy as np
import pandas as pd

conn = hive.Connection(host=host_name, port=port, username=user,
                       password=password, database=database, auth='CUSTOM')


def select_query(conn, table):
    cur = conn.cursor()
    cur.execute('desc {}'.format(table))
    result = cur.fetchall()
    cols = [r[0] for r in result]
    cur.execute('select * from  {}'.format(table))
    result = cur.fetchall()
    df = pd.DataFrame(result, columns=cols)
    return df


# houses = select_query(conn, 'houses')
# print(houses.head())

#-----------------------------------------------

def hive_percentiles_aggregate(conn, input_table, output_table, group,
                               var, pctl_from, pctl_to, pctl_by, pref = 'P_'):
    cur = conn.cursor()
    pctl_range = np.arange(pctl_from, pctl_to + pctl_by, pctl_by)
    pctl_range = pctl_range[pctl_range <= 100]
    hv_arr = 'array' + str(tuple(r / 100 for r in pctl_range))

    cur.execute("drop table quantiles_00000")
    qr = """ create temporary table quantiles_00000 as 
             select 
                {group}, percentile(cast( {var} as bigint), {hv_arr}) as pctls 
             from 
                {table}
             group by {group}
        """.format(var=var, group=group, hv_arr=hv_arr,
                   table=input_table)
    cur.execute(qr)
    arr = ', '.join(['pctls[{}] as {}{}'.format(i, pref, r)
                     for i, r in enumerate(pctl_range)])
    cur.execute("drop table {}".format(output_table))
    qr = """ 
            create  table {output_table} stored as orc as 
            select 
                {group}, {arr} 
            from 
                quantiles_00000
         """.format(output_table = output_table, arr = arr, group=group)
    cur.execute(qr)

    res = select_query(conn, 'percentiles_calc')
    return res


out = hive_percentiles_aggregate(conn, input_table = 'houses',
                        output_table= 'house_perc', group='animal', var='total',
                        pctl_from=1 , pctl_to=100, pctl_by=1, pref = 'P_')


print(out)

# cur = conn.cursor()
# group  = 'animal'
# var ='total'
# pref = 'P_'
#
# pctl_from, pctl_to, pctl_by = 0, 100, 5
# pctl_range = np.arange(pctl_from, pctl_to + pctl_by, pctl_by)
# pctl_range = pctl_range[pctl_range<=100]
# pctl_list = tuple(r/100 for r in pctl_range)
# hv_arr = 'array'+str(pctl_list)
#
# cur.execute("drop table quantiles")
# qr = """ create temporary table quantiles as
#          select
#             {group}, percentile(cast( {var} as bigint), {hv_arr}) as pctls
#          from
#             {table}
#          group by {group}
#     """.format(var=var, group=group, hv_arr=hv_arr, table='houses')
# cur.execute(qr)
#
#
# arr = ', '.join( ['pctls[{}] as {}{}'.format(i,pref,r)
#                     for i,r in enumerate(pctl_range)] )
# cur.execute("drop table percentiles_calc")
# qr = """
#         create  table percentiles_calc as
#         select
#             {group}, {}
#         from quantiles
#      """.format(arr, group=group)
# cur.execute(qr)
#
# df2 = select_query(conn, 'percentiles_calc')
# print(df2)




















