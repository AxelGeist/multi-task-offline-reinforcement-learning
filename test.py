import pandas as pd
from io import StringIO

def addSizeToCSV(csv_path: str):
    dataset_size = csv_path.split('_')[-2]
    
    df = pd.read_csv(csv_path)

    # Insert a new column with the value 40 between 'Dataset' and 'Environment'
    df.insert(df.columns.get_loc('Environment'), 'Size', dataset_size)

    # Save the updated DataFrame back to a CSV file
    df.to_csv(csv_path, index=False)
    
    
    
addSizeToCSV('models/sac_bc\optimal_40_10/results.csv')
addSizeToCSV('models/sac_bc\optimal_40_11/results.csv')
addSizeToCSV('models/sac_bc\optimal_40_12/results.csv')
addSizeToCSV('models/sac_bc\optimal_40_13/results.csv')
addSizeToCSV('models/sac_bc\optimal_40_14/results.csv')

addSizeToCSV('models/sac_bc\suboptimal_80_10/results.csv')
addSizeToCSV('models/sac_bc\suboptimal_80_11/results.csv')
addSizeToCSV('models/sac_bc\suboptimal_80_12/results.csv')
addSizeToCSV('models/sac_bc\suboptimal_80_13/results.csv')
addSizeToCSV('models/sac_bc\suboptimal_80_14/results.csv')

addSizeToCSV('models/sac_bc\mixed_80_10/results.csv')
addSizeToCSV('models/sac_bc\mixed_80_11/results.csv')
addSizeToCSV('models/sac_bc\mixed_80_12/results.csv')
addSizeToCSV('models/sac_bc\mixed_80_13/results.csv')
addSizeToCSV('models/sac_bc\mixed_80_14/results.csv')


