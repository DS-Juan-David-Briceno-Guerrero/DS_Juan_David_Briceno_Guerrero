#Define a function to transform a modis date into a calendar date.
def convert_date_to_days_since_yyyy_mm_dd(date, c):
    y, m, d = c 
    days_since_yyyy_mm_dd = date - datetime(year=y, month=m, day=d)
    days_since_yyyy_mm_dd = days_since_yyyy_mm_dd.days + 1
    return days_since_yyyy_mm_dd

#Define a function that obtain days from specified parameters.
def convert_days_since_yyyy_mm_dd_to_date(int_days, c):
    y, m, d = c
    int_days_mod = int_days -1
    date = datetime(year=y, month=m, day=d) + timedelta(days = int_days_mod)
    return date

#Define a function to transform a modis date into a calendar date.
def convert_modis_date_to_calendar_date(str_modis_date):
    y, d = str_modis_date[1:5], str_modis_date[6:len(str_modis_date)]
    date = convert_days_since_yyyy_mm_dd_to_date(int(d), (int(y),1,1))
    date = date.strftime("%Y-%m-%d")
    return date

#Define a function to transform a calendar date into modis date.
def convert_calendar_date_to_modis_date(str_calendar_date):
    #Convert into datetime.
    date = datetime.strptime(str_calendar_date, '%Y-%m-%d')
    
    #Obtains days since begining of year.
    days = convert_date_to_days_since_yyyy_mm_dd(date,(date.year,1,1))
    
    #Return modis date.
    if days<=99:
        modis_date = 'A'+str(date.year)+'0'+str(days)
    else:
        modis_date = 'A'+str(date.year)+str(days)
    #Return.
    return(modis_date)

#Defines a function that allows to execute a bash command. from python.
def bash_command(cmd):
    subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    
    
    
#Defines a function to obtain links for a single period.
#Obtain links for several periods.
def modis_data_obtain_rasters_links_single_period(product,instrument,period, selected_tiles):
    #Obtain soup.
    req = Request("https://e4ftl01.cr.usgs.gov/"+product+"/"+instrument+"/"+period)
    html_page = urlopen(req)
    soup = BeautifulSoup(html_page, "lxml")
    
    #Stage 1.
    links = []
    for link in soup.findAll('a'):
        links.append(link.get('href'))
    
    #Stage 2.
    #Obtain only links related to hdf images.
    leng = len(links)
    cond = True
    while cond==True:
        #Remove elements.
        for w in links:
            if('hdf' not in w): 
                links.remove(w)
        #Check conditions.
        if(len(links)<leng):
            cond = True
            leng = len(links)
        else:
            cond = False
    
    #Stage 3.
    #Remove links with the '.xml' part.
    links_new = []
    for i in links:
        if('.xml' not in i):
            links_new.append(i)
    links = links_new
    
    #Stage 4.
    #Remove duplicates.
    links = list(dict.fromkeys(links))
    
    #Stage 5.
    #Links from selected tiles.
    selected_tiles = selected_tiles
    links_selected_tiles = []
    for j in selected_tiles:
        links_selected_tiles.append([x for x in links if j in x])
    links_selected_tiles = new_d = [i[0] for i in links_selected_tiles]
    
    #Stage 6.
    #Complete links to download.
    links_complete_path=[]
    for i in links_selected_tiles:
        links_complete_path.append("https://e4ftl01.cr.usgs.gov/"+product+"/"+instrument+"/"+period+"/"+i)
    
    #Return.
    return links_complete_path


#Should be paralelized.
def modis_data_obtain_rasters_links_multiple_periods(product, instrument, periods, selected_tiles):
    all_links = []
    
    #Append of links.
    for i in periods:
        print(i)
        l = modis_data_obtain_rasters_links_single_period(product, instrument, i, selected_tiles)
        all_links.append(l)
        
    #Flatten list.
    all_links = [j for i in all_links for j in i]
    
    #return.
    return(all_links)