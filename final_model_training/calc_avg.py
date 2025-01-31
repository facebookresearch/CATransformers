import csv
import pandas as pd
def calculate_average(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    # Calculate the average of "text_retrieval_recall@1" and "image_retrieval_recall@1"
    df['average_recall'] = (df['text_retrieval_recall@1'] + df['image_retrieval_recall@1']) / 2
    # Create a new column 'value' which is either 'acc1', 'average_recall', or 'mean_average_precision'
    df['value'] = df.apply(lambda row: row['acc1'] if not pd.isnull(row['acc1']) else 
                           (row['average_recall'] if not pd.isnull(row['average_recall']) else 
                            row['mean_average_precision']), axis=1)
    # Group by model and calculate the total average value
    total_average_per_model = df.groupby('model')['value'].mean().reset_index()
    # Group by model and task, and calculate the average value
    average_values_per_model_per_task = df.groupby(['model', 'task'])['value'].mean().reset_index()
    # Pivot the data to get the desired format
    pivoted_data = df.pivot(index='model', columns='dataset', values='value')
    return total_average_per_model, average_values_per_model_per_task, pivoted_data
# Specify the CSV file path
csv_file_path = ''
# Call the function to calculate the average values
total_average_per_model, average_values_per_model_per_task, pivoted_data = calculate_average(csv_file_path)
# Print the results
print("Total Average per Model:")
print(total_average_per_model)
print("\nAverage Values per Model per Task:")
print(average_values_per_model_per_task)

