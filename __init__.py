from orcamento_mg import *
from scriptAgrupamento import *
from treat_datasus import *

if __name__ == "__main__":
    consolidate_datasus()
    create_splited_columns()
    exit(0)