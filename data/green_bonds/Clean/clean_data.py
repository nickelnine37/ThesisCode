import pandas as pd

import os

base = 'Muni Green Bond data Yields_FINAL_v2/Muni Green Bond data Yields/Muni Green Bond data Yields/Green Bonds- Muni'
regions = os.listdir(base)

all_data = {}
features = pd.DataFrame()


for region in regions:
        
    for period in ['Issuedate=2000 to 2015', 'Issue date= 2015 to 2021']:
        
        for activity in ['Active', 'Matured']:
            
            print(f'/{region}/{period}/{activity}/')
            
            folder = base + f'/{region}/{period}/{activity}/Bonds based on the year of issuance'
    
            files = sorted(os.listdir(folder))
        
            for f in files:
                
                if 'Active' in f:
                    
                    df = pd.read_excel(os.path.join(folder, f), sheet_name=None)
                    
                    if list(df.keys()) == ['Sheet1']:
                        
                        ddf = pd.read_excel(os.path.join(folder, f.replace('Active', '')))
                        
                        name = ddf['ID_CUSIP'][0]
                                                
                        df[name] = df.pop('Sheet1')
                        
                    keys = list(df.keys())
                    
                    for ticker in keys:
                        
                        df[ticker.replace('Muni', '').strip()] = df.pop(ticker)
                    
                    all_data = all_data | df
                    
                else:
                    
                    ddf = pd.read_excel(os.path.join(folder, f))
                    ddf['region'] = region
                                        
                    features = pd.concat([features, ddf])
                    
                    
features = features.drop_duplicates(subset='ID_CUSIP').set_index('ID_CUSIP')

yields = pd.concat([all_data[code].set_index('date').rename({'YLD_YTM_MID': code}, axis=1) for code in all_data.keys()], axis=1).sort_index()

bonds = sorted(list(set(yields.columns.tolist()).intersection(features.index.tolist())))

yields = yields[bonds]
features = features.loc[bonds]


feat_path = 'Muni Green Bond Data Features_FINAL_v2/Muni Green Bond Data Features/Region-by-Region'

issuers = pd.concat([pd.read_excel(os.path.join(feat_path, f), header=None).assign(Region=f.replace('-Stat.xlsx', '')) for f in os.listdir(feat_path) if 'Stat' in f], axis=0)

issuers = issuers[~issuers[0].str.contains('Name of the issuer')]

issuers = issuers.rename({0: 'Issuer', 1: 'First date issued GB', 2: 'Last date issued GB', 3: 'Total amount  (sum)', 4: 'Number of bonds'}, axis=1)


yields.to_csv('Yields.csv')

features.to_csv('Features.csv')

issuers.to_csv('Issuers.csv')
