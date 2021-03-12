from typing import Callable

from script_agrupamento import *
from treat_datasus import *
from orcamento_mg import *
from data_linkage import *
from analise_geral import *
from analise_por_ano import *
###
# Configure logs
###
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=FORMAT)


###
# END - Configure logs
###

def run_routine(data: str, routine: Callable):
    log.info(f'Consolidating {data} Files')
    routine()
    log.info(f'{data} Files Consolidated')

def run_analise(data, routine: Callable):
    log.info(f'Analyzing data: {data}')
    processes = [multiprocessing.Process(target=routine, args=(i,),daemon=False) for i in data]
    [res.start() for res in processes]
    [res.join() for res in processes]
    log.info(f'Final data analizys')
    


if __name__ == "__main__":
    log.info('Initializing routines')

    processes = [multiprocessing.Process(target=run_routine, args=('Data Sus', consolidate_datasus,),daemon=False)
                ,multiprocessing.Process(target=run_routine, args=("Data Budget", consolidate_budget,),daemon=False)]

    [res.start() for res in processes]
    
    [res.join() for res in processes]

    
    log.info('Consolidating Budgeting')
    create_splited_columns()
    log.info('Budget Consolidated')
    import_files()

    anos = [2009,2010,2011,2012,2013,2014,2015,2017,2018]
    run_analise(anos, analise_anual)
    # run_analise(anos, analise_reg_por_ano)
    analise_main()
    
    exit(0)
