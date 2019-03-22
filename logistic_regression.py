import os
import sys
import pandas
import time
import json

from pytaskmanager.node.FlaskIO import ClientContainerProtocol

# loggers
info = lambda msg: sys.stdout.write("info > "+msg+"\n")
warn = lambda msg: sys.stdout.write("warn > "+msg+"\n")

def master(token, columns, decimal, seperator):
    """Master algoritm.
    
    Algorithm which request the dsummary from all sites and then
    computes the overal (whereas possible) of the entire (distributed)
    dataset. This is simalar what the researcher would do without a
    master container.

    Keyword arguments:
    token -- JWT token to access the central server
    columns -- column names/type pairs
    decimal -- how decimals are seperated in csv-file, usually "." or ","
    seperator -- line seperator in csv-file, usually ";" or ":"
    """

    # post task to all nodes in collaboration
    info("Setup server communication client")
    client = ClientContainerProtocol(
        token=token, 
        host=os.environ["HOST"],
        port=os.environ["PORT"], 
        path=os.environ["API_PATH"]
    )
       
    # define the input for the summary algorithm
    info("Defining input paramaeters")
    input_ = {
        "method": "summary",
        "args": [],
        "kwargs": {
            "decimal": decimal, 
            "seperator":seperator, 
            "columns":columns
        }
    }

    # collaboration and image is stored in the key, so we do not need
    # to specify these
    info("Creating node tasks")
    task = client.create_new_task(input_)

    # wait for all results
    # TODO subscribe to websocket, to avoid polling
    task_id = task.get("id")
    task = client.request(f"task/{task_id}")
    while not task.get("complete"):
        task = client.request(f"task/{task_id}")
        info("Waiting for results")
        time.sleep(1)

    info("Obtaining results")
    results = client.get_results(task_id=task.get("id"))
    results = [json.loads(result.get("result")) for result in results]

    # for x in results:
    #     info(str(x))

    # process the output 
    info("Process node info to global stats")
    g_stats = {}

    # check that all dataset reported their headers are correct
    info("Check if all column names on all sites are correct")
    g_stats["column_names_correct"] = all([x["column_names_correct"] for x in results])
    # info(f"correct={g_stats['column_names_correct']}")

    # count the total number of rows of all datasets
    info("Count the total number of all rows from all datasets")
    g_stats["number_of_rows"] = sum([x["number_of_rows"] for x in results])
    # info(f"n={g_stats['number_of_rows']}")
    
    # compute global statics for numeric columns
    info("Computing numerical column statistics")
    columns_series = pandas.Series(columns)
    numeric_colums = columns_series.loc[columns_series.isin(['Int64','float64'])]
    for header in numeric_colums.keys():

        n = g_stats["number_of_rows"]
        
        # extract the statistics for each column and all results
        stats = [ result["statistics"][header] for result in results ] 
        
        # compute globals
        g_min = min([x.get("min") for x in stats])
        # info(f"g_min={g_min}")
        g_max = max([x.get("max") for x in stats])
        # info(f"g_max={g_max}")
        g_nan = sum([x.get("nan") for x in stats])
        # info(f"g_nan={g_nan}")
        g_mean = sum([x.get("sum") for x in stats]) / (n-g_nan)
        # info(f"g_mean={g_mean}")
        g_std = (sum([x.get("sq_dev_sum") for x in stats]) / (n-1-g_nan))**(0.5)
        
        # estimate the median
        # see https://stats.stackexchange.com/questions/103919/estimate-median-from-mean-std-dev-and-or-range
        u_std = (((n-1)/n)**(0.5)) * g_std 
        g_median = [
            max([g_min, g_mean - u_std]),
            min([g_max, g_mean + u_std])
        ]
        
        g_stats[header] = {
            "min": g_min,
            "max": g_max,
            "nan": g_nan,
            "mean": g_mean,
            "std": g_std,
            "median": g_median
        }

        #TODO process categorial columns
    info("master algorithm complete")

    return g_stats

def summary(columns, decimal, seperator):
    """Node algorithm to compute site-statistics.
    
    Algorithm which computes a summary (min,max,avg, etc) from all sites

    Keyword arguments:
    columns -- column names/type pairs
    decimal -- how decimals are seperated in csv-file, usually "." or ","
    seperator -- line seperator in csv-file, usually ";" or ":"
    """

    # create series from input column names
    columns_series = pandas.Series(data=columns)

    # create pandas dataframe from csv-file
    info("Reading database-file")
    dataframe = pandas.read_csv(
        os.environ['DATABASE_URI'], 
        sep=seperator,
        decimal=decimal,
        dtype=columns
    )

    # compare column names from dataset to the input column names
    info("Checking column-names")
    column_names_correct = list(dataframe.keys()) == list(columns_series.keys())
    if not column_names_correct:
        warn("Column names do not match. Exiting.")
        return {"column_names_correct": column_names_correct}

    # count the number of rows in the dataset
    # TODO should the minimal row-count be controlled by the Node?
    info("Counting number of rows")
    number_of_rows = len(dataframe)
    if number_of_rows < 10:
        warn("Dataset has less than 10 rows. Exiting.")
        return {
            "column_names_correct": column_names_correct,
            "number_of_rows": number_of_rows
        }

    # min, max, median, average, Q1, Q3, missing_values
    columns = {}
    numeric_colums = columns_series.loc[columns_series.isin(['Int64','float64'])]
    for column_name in numeric_colums.keys():
        info(f"Numerical column={column_name} is processed")
        column_values = dataframe[column_name]
        q1, median, q3 = column_values.quantile([0.25,0.5,0.75]).values
        mean = column_values.mean()
        minimum = column_values.min()
        maximum = column_values.max()
        nan = column_values.isna().sum()
        total = column_values.sum()
        std = column_values.std()
        sq_dev_sum = (column_values-mean).pow(2).sum()
        columns[column_name] = {
            "min": minimum,
            "q1": q1,
            "median": median,
            "mean": mean,
            "q3": q3,
            "max": maximum,
            "nan": int(nan),
            "sum": total,
            "sq_dev_sum": sq_dev_sum,
            "std": std
        }
        
    # return the categories in categorial columns
    categoral_colums = columns_series.loc[columns_series.isin(['category'])]
    for column_name in categoral_colums.keys():
        info(f"Categorial column={column_name} is processed")
        columns[column_name] = dataframe[column_name].value_counts().to_dict()

    return {
        "column_names_correct": column_names_correct,
        "number_of_rows": number_of_rows,
        "statistics": columns
    }
