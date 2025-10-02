# Organização diretório ./data/
### raw_data/
Pasta com arquivos referentes à base de dados sem qualquer tratamento.
- **Dodgers.data:** conjunto de dados que representa a medição, pelo sensor de loop, de carros passando na rampa de acesso para a Rodovia 101 Norte. As medições contêm data, horário e contagem de carros capturados no momento.
- **Dodgers.event:** conjunto de dados que representa os jogos do Dodgers 12/04/2005 e 29/09/2005. Cada linha representa data, horário de início do jogo, horário de fim do jogo, número de torcedores que compareceram ao evento, time adversário e resultado do jogo (Won/Lost e pontuação).
- **Dodgers.name:** explicação geral dos datasets, com informações como número de observações, formatação dos dados e notas importantes.
- **dodgers+loop+sensor.zip:** arquivo zip com todos os datasets iniciais.

### Dodgers_data.csv
CSV com os dados de Dodgers.data devidamente tratados. Não é necessário para o projeto em si, só foi utilizado para a criação do arquivo de dados principal *(Dodgers_processed.csv)*.

### Dodgers_events.csv
CSV com os dados de Dodgers.events tratados. Foi utilizado para marcar as medições ocorridas durante um evento do Dodgers no arquivo de dados principal. Também não é necessário para o projeto em si.

### Dodgers_processed.csv
Dataset principal do projeto. De modo geral, é constituído pelo arquivo Dodger_data.csv com um coluna extra, marcando as medições realizadas (1) ou não (0) durante a ocorrência de um jogo do Dodgers.

### process_data.py
Script de tratamento dos dados. As alterações principais realizadas foram:
1. Tratamento de valores -1 para a contagem de carros. Para resolver isso, foi feita a média aritmética dos 4 valores válidos mais próximos.
2. Adição da coluna booleana 'Event', simbolizando se tall medição foi feita durante a duração de um jogo ou não.  