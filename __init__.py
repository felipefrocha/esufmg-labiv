from typing import Callable

from scriptAgrupamento import *
from treat_datasus import *
from orcamento_mg import *
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


if __name__ == "__main__":
    log.info('Initializing routines')

    processes = [multiprocessing.Process(target=run_routine, args=('Data Sus', consolidate_datasus,),daemon=False)
                ,multiprocessing.Process(target=run_routine, args=("Data Budget", consolidate_budget,),daemon=False)]
    [res.start() for res in processes]
    [res.join() for res in processes]
    log.info('Consolidating Budgeting')
    create_splited_columns()
    log.info('Budget Consolidated')
    exit(0)
