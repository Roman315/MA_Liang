import pandas as pd

def export_alt(path_export, list_final):

    
    exportpath = path_export + r'\\Results.xlsx' #
    
    writer = pd.ExcelWriter(exportpath, engine='xlsxwriter') # Create a Pandas Excel writer using XlsxWriter as the engine
    workbook  = writer.book
    bold = workbook.add_format({'bold': True})
    
    #%% sheet: Allgemein
    
    #setup
    worksheet1 = workbook.add_worksheet('Allgemein')
    worksheet1.write('A1', 'Dateiname', bold)
    worksheet1.write('B1', 'Tiefe', bold)
    worksheet1.write('C1', 'Durchmesser', bold)
    worksheet1.set_column(2, 2, len('Durchmesser'))
    
    #fill with data
    for n_analyzes in range(0, len(list_final)):
        filename = list_final[n_analyzes][0][:31] #cut filename at 31 characters as this is the limitation for excel worksheet names
        res_depth = list_final[n_analyzes][1]
        res_diameter = list_final[n_analyzes][2]
        df_output = list_final[n_analyzes][3]
    
        worksheet1.write('A' + str(n_analyzes + 2), filename)
        worksheet1.set_column(0, 0, len(filename))
        
        worksheet1.write('B' + str(n_analyzes + 2), -res_depth)
        worksheet1.write('C'+ str(n_analyzes + 2), res_diameter)
    
        #add chart
        '''
        chart = workbook.add_chart({'type': 'column'}) # Create a chart object
        chart.add_series({'values': '=Allgemein!$B$' + str(n_analyzes + 2) + ':$C$' + str(n_analyzes + 2)}) # Configure the series of the chart from the dataframe data
        chart.set_y_axis({'name': 'µm'})
        worksheet1.insert_chart(1, 6 + n_analyzes * 8, chart) # Insert the chart into the worksheet
        '''
    
    # sheet: Details
        df_output.to_excel(writer, filename) # Convert the dataframe to an XlsxWriter Excel object
        
        worksheet2 = writer.sheets[filename]
        
        worksheet2.set_column(1, 1, len('Lochnummer '))
        worksheet2.set_column(2, 2, len('Durchmesser'))
        worksheet2.set_column(3, 3, len('Std_Durchmesser'))
        worksheet2.set_column(4, 4, len('Std_Tiefe'))

        # Close the Pandas Excel writer and output the Excel file
    writer.save()

def exportlumentum(path_export, list_final):
    results = pd.DataFrame(columns=['Nr','Wavelength / nm','PRR / kHz','NPulse','EPulse / uJ','Diameter / um','Diameter std / um','Depth / um','Depth std / um','Data size / px', 'Run time wls / s','Run time hole numbering / s','Run time geometric center / s', 'Run time total / s']) #create dataframe
    for j in range(len(list_final)):
        lst = list_final[j][0] #copy filename string into new variable
        for unit in ['nm','kHz','ppD','uJ','_Höhe']:
            lst = lst.replace(unit,'') #remove units from filename string
        lst = lst.replace(',','.') #convert , to . in filename string
        lst = lst.split('_') #split string into list of substrings containing 'Nr','Wavelength/nm','PRR/kHz','Npulse','EPulse/uJ'
        lst.append(list_final[j][2]) #append diameter to list
        lst.append(list_final[j][5]) #append diameter std to list
        lst.append(-list_final[j][1]) #append depth to list
        lst.append(list_final[j][4]) #append depth std to list
        for k in [6,7,8,9,10]:
            lst.append(list_final[j][k]) #append data size and runtimes to list
        results.loc[len(results)] = lst #append list to pandas dataframe
    results.to_csv(path_export + r'\\Results_Lumentum.csv',sep='\t') #save to csv
    results.to_pickle(path_export + r'\\Results_Lumentum.pkl') #save to pkl
    
def export(path_export, list_final, n_resize):
    results = pd.DataFrame(columns=['Name','Diameter / um','Diameter std / um','Depth / um','Depth std / um','Data size / px', 'Run time wls / s','Run time hole numbering / s','Run time geometric center / s', 'Run time total / s']) #create dataframe
    for j in range(len(list_final)):
        lst = []
        lst.append(list_final[j][0]) #copy filename string into new va
        lst.append(list_final[j][2]) #append diameter to list
        lst.append(list_final[j][5]) #append diameter std to list
        lst.append(-list_final[j][1]) #append depth to list
        lst.append(list_final[j][4]) #append depth std to list
        for k in [6,7,8,9,10]:
            lst.append(list_final[j][k]) #append data size and runtimes to list
        results.loc[len(results)] = lst
    results.to_csv(path_export+r'\Results_n_'+str(n_resize)+'.csv') #save to csv
    results.to_pickle(path_export+r'\Results_n_'+str(n_resize)+'.pkl') #save to pkl