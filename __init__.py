from scriptAgrupamento import *
from treat_datasus import *

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

if __name__ == "__main__":
    log.info('Initializing routines')
    log.info('Consolidating Data Sus Files')
    consolidate_datasus()
    log.info('Data Sus Files Consolidated')
    log.info('Consolidating Budgeting')
    create_splited_columns()
    log.info('Budget Consolidated')
    exit(0)
