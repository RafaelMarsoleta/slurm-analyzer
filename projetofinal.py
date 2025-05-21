import pandas as pd
import matplotlib.pyplot as plt

#Classe para análise de dados do Slurm
class SlurmAnalyzer:
    
    #Construtor da classe
    def __init__(self, job_file: str, user_file: str): 
        self.job_file = job_file
        self.user_file = user_file
        self.jobs = None
        self.users = None
    
    #Carrega e processa os dados
    def loadProcess(self) -> None:
        self._load_data()
        self._convert_timestamps()
        self._compute_time_jobs()
        print("Processamento concluido!")
    
    # Método para carregar os dados dos arquivos CSV
    def _load_data(self) -> None:
        self.jobs = pd.read_csv(self.job_file)
        self.users = pd.read_csv(self.user_file)

        if 'id_user' in self.jobs.columns and 'user_name' in self.users.columns:
            self.jobs = self.jobs.merge(
                self.users[['id_user', 'user_name']],
                how='left',
                on='id_user'
            )
    
    #Conversão de timestamps para formato datetime
    def _convert_timestamps(self) -> None:        
        datetime_cols = ['time_submit', 'time_eligible', 'time_start', 'time_end', 'mod_time']
        for col in datetime_cols:
            if col in self.jobs.columns:
                self.jobs[col] = pd.to_datetime(self.jobs[col], unit='s', errors='coerce')
    
    # Cálculo de tempo de execução e tempo de espera dos jobs
    def _compute_time_jobs(self) -> None:
        self.jobs['job_duration_min'] = (self.jobs['time_end'] - self.jobs['time_start']).dt.total_seconds() / 60
        self.jobs['wait_time_min'] = (self.jobs['time_start'] - self.jobs['time_submit']).dt.total_seconds() / 60
    
    #Analisa jobs por usuário usando dados pré-processados.
    def analyzeJobsUser(self, top_n: int = 10) -> pd.DataFrame:
        
        if not isinstance(self.jobs, pd.DataFrame):
            raise ValueError("Dados não carregados. Execute loadProcess() primeiro.")
        
        result = (self.jobs['user_name']
                 .value_counts()
                 .reset_index()
                 .rename(columns={'index': 'Usuário', 'user_name': 'Total de Jobs'}))
        
        return result.head(top_n)
    
    #Gráfico de barras comparando uso de CPU e RAM por usuário.
    def cpuRamUsage(self, top_n: int = 10):
        
        if not all(col in self.jobs.columns for col in ['cpus_req', 'mem_req', 'user_name']):
            raise ValueError("Colunas de CPU, RAM ou usuário não encontradas")
            
        usage = self.jobs.groupby('user_name').agg({
            'cpus_req': 'sum',
            'mem_req': 'sum'
        }).nlargest(top_n, 'cpus_req')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        usage.plot(kind='bar', ax=ax, secondary_y=['mem_req'])
        ax.set_title(f'Top {top_n} Usuários por Uso de Recursos')
        ax.set_ylabel('CPU Total Requisitada')
        plt.ylabel('RAM Total Requisitada (GB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    #Gráfico de linha com evolução mensal do top 5 usuários.
    def clusterUsers(self):
        
        if 'time_submit' not in self.jobs.columns:
            raise ValueError("Dados de tempo não encontrados")
            
        top_users = self.jobs['user_name'].value_counts().nlargest(5).index
        monthly_data = self.jobs[self.jobs['user_name'].isin(top_users)].copy()
        monthly_data['month'] = monthly_data['time_submit'].dt.to_period('M')
        
        plt.figure(figsize=(12, 6))
        for user in top_users:
            user_data = monthly_data[monthly_data['user_name'] == user]
            user_data.groupby('month').size().plot(label=user, marker='o')
        
        plt.title('Evolução Mensal do Top 5 Usuários')
        plt.xlabel('Mês')
        plt.ylabel('Quantidade de Jobs')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    #Gráfico de linha com jobs por dia.
    def jobsDay(self):
        if 'time_submit' not in self.jobs.columns:
            raise ValueError("Dados temporais não encontrados")
            
        daily_jobs = self.jobs.set_index('time_submit').resample('D').size()
        
        plt.figure(figsize=(12, 6))
        daily_jobs.plot()
        plt.title('Jobs Submetidos por Dia')
        plt.xlabel('Data')
        plt.ylabel('Quantidade de Jobs')
        plt.grid()
        plt.tight_layout()
        plt.show()


def main():
    analyzer = SlurmAnalyzer("ppgi_job_table.csv", "user_table.csv")
    analyzer.loadProcess()
    
    # Gerar todos os gráficos
    analyzer.cpuRamUsage(top_n=5)
    analyzer.clusterUsers()
    analyzer.jobsDay()
    
   

if __name__ == "__main__":
    main()