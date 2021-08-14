from typing import Callable
import nomad

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
    log.info(f'Final data analysis')
    
    

def launch_job():

    job = {'Job': {'AllAtOnce': None,
	  'Constraints': None,
	  'CreateIndex': None,
	  'Datacenters': ['dc1'],
	  'ID': 'example',
	  'JobModifyIndex': None,
	  'Meta': None,
	  'ModifyIndex': None,
	  'Name': 'example',
	  'Namespace': None,
	  'ParameterizedJob': None,
	  'ParentID': None,
	  'Payload': None,
	  'Periodic': None,
	  'Priority': None,
	  'Region': None,
	  'Stable': None,
	  'Status': None,
	  'StatusDescription': None,
	  'Stop': None,
	  'SubmitTime': None,
	  'TaskGroups': [{'Constraints': None,
	    'Count': 1,
	    'EphemeralDisk': {'Migrate': None, 'SizeMB': 300, 'Sticky': None},
	    'Meta': None,
	    'Name': 'cache',
	    'RestartPolicy': {'Attempts': 10,
	     'Delay': 25000000000,
	     'Interval': 300000000000,
	     'Mode': 'delay'},
	    'Tasks': [{'Artifacts': None,
	      'Config': {'image': 'redis:3.2', 'port_map': [{'db': 6379}]},
	      'Constraints': None,
	      'DispatchPayload': None,
	      'Driver': 'docker',
	      'Env': None,
	      'KillTimeout': None,
	      'Leader': False,
	      'LogConfig': None,
	      'Meta': None,
	      'Name': 'redis',
	      'Resources': {'CPU': 500,
	       'DiskMB': None,
	       'IOPS': None,
	       'MemoryMB': 256,
	       'Networks': [{'CIDR': '',
		 'Device': '',
		 'DynamicPorts': [{'Label': 'db', 'Value': 0}],
		 'IP': '',
		 'MBits': 10,
		 'ReservedPorts': None}]},
	      'Services': [{'AddressMode': '',
		'CheckRestart': None,
		'Checks': [{'Args': None,
		  'CheckRestart': None,
		  'Command': '',
		  'Header': None,
		  'Id': '',
		  'InitialStatus': '',
		  'Interval': 10000000000,
		  'Method': '',
		  'Name': 'alive',
		  'Path': '',
		  'PortLabel': '',
		  'Protocol': '',
		  'TLSSkipVerify': False,
		  'Timeout': 2000000000,
		  'Type': 'tcp'}],
		'Id': '',
		'Name': 'global-redis-check',
		'PortLabel': 'db',
		'Tags': ['global', 'cache']}],
	      'ShutdownDelay': 0,
	      'Templates': None,
	      'User': '',
	      'Vault': None}],
	    'Update': None}],
	  'Type': 'service',
	  'Update': {'AutoRevert': False,
	   'Canary': 0,
	   'HealthCheck': None,
	   'HealthyDeadline': 180000000000,
	   'MaxParallel': 1,
	   'MinHealthyTime': 10000000000,
	   'Stagger': None},
	  'VaultToken': None,
	  'Version': None}}

    my_nomad = nomad.Nomad(host='172.29.189.47')

    response = my_nomad.job.register_job("example", job)
    
    read_job = my_nomad.job.get_job("example")


if __name__ == "__main__":
    log.info('Initializing routines')

    processes = [multiprocessing.Process(target=run_routine, args=('Data Sus', launch_job,),daemon=False)]
                # ,multiprocessing.Process(target=run_routine, args=("Data Budget", consolidate_budget,),daemon=False)]

    [res.start() for res in processes]
    
    [res.join() for res in processes]

    
    #log.info('Consolidating Budgeting')
    #create_splited_columns()
    #log.info('Budget Consolidated')
    #import_files()

    #anos = [2009,2010,2011,2012,2013,2014,2015,2017,2018]
    #run_analise(anos, analise_anual)
    # run_analise(anos, analise_reg_por_ano)
    #analise_main()
    
    exit(0)
