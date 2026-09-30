# EXP-001 — Baseline do agente

## Objetivo

Estabelecer uma linha de base para avaliar o comportamento inicial do agente
evoluído por NEAT antes da introdução de otimizações de desempenho e de
informações espaciais adicionais sobre o ambiente.

## Configuração

- População: 50
- Gerações: 50
- Inputs: 37
- Outputs: 7
- Arquitetura: Feed-forward
- Função de ativação: tanh
- Conexões iniciais: full_direct
- Seed do ambiente: 42
- Avaliação: sequencial
- Máximo de steps: 10.000
- Limite de estagnação: 180 steps
- Checkpoint: a cada 5 gerações

## Função de fitness

fitness = progresso_horizontal
        + 10.000, se concluir a fase
        - 100, se morrer

Onde:

progresso_horizontal = maior_x - x_inicial

## Métricas coletadas

- Melhor fitness
- Fitness médio
- Desvio padrão do fitness
- Número de espécies
- Número de nós
- Número de conexões
- Tempo por geração

## Resultado observado

O agente foi capaz de aprender progressão horizontal e apresentar
comportamentos básicos de sobrevivência.

Entretanto, durante a avaliação visual do melhor indivíduo, o agente
permaneceu bloqueado diante de um cano, demonstrando dificuldade para
reagir a obstáculos estáticos do cenário.

## Interpretação

Os 37 sensores utilizados forneciam informações sobre o estado do Mario
e sobre inimigos, mas não forneciam uma representação espacial explícita
da geometria do cenário.

O resultado estabeleceu o baseline para os experimentos seguintes.


---

# EXP-002 — Avaliação paralela com multiprocessing

## Objetivo

Avaliar a utilização de processamento paralelo na avaliação dos genomas,
reduzindo o tempo necessário para execução das gerações sem alterar a
representação sensorial do agente.

## Configuração

- População: 50
- Gerações: 50
- Inputs: 37
- Outputs: 7
- Arquitetura: Feed-forward
- Função de ativação: tanh
- Conexões iniciais: full_direct
- Seed do ambiente: 42
- Avaliação: paralela
- Workers: 12
- Máximo de steps: 10.000
- Limite de estagnação: 180 steps
- Checkpoint: a cada 5 gerações

## Função de fitness

Mantida igual ao EXP-001:

fitness = progresso_horizontal
        + 10.000, se concluir a fase
        - 100, se morrer

## Resultado

- Tempo de treinamento: 459,10 s
- Melhor fitness: 760
- Posição inicial X: 40
- Maior posição X: 900
- Distância percorrida: 860
- Steps executados: 772
- Reward acumulado do ambiente: 810
- Fase concluída: Não
- Estado final: Morte

## Comportamento observado

Durante a avaliação visual do melhor indivíduo:

- o agente aprendeu a avançar pela fase;
- alcançou um obstáculo estático relevante;
- não conseguiu transpor o cano;
- continuou tentando avançar contra o obstáculo;
- posteriormente morreu devido à interação com um inimigo.

## Interpretação

O multiprocessing alterou o processo computacional de avaliação dos
genomas, mas não adicionou novas informações à percepção do agente.

Consequentemente, permaneceu a limitação observada no EXP-001 relacionada
à percepção da geometria do cenário.

## Hipótese para o próximo experimento

Se for fornecida ao NEAT uma representação espacial compacta do ambiente
ao redor do Mario, o processo evolutivo poderá desenvolver comportamentos
capazes de reagir a obstáculos estáticos, como canos e buracos.


# EXP-003 — Percepção espacial 5×5

## Objetivo

Avaliar o impacto da adição de informações espaciais do cenário sobre
a capacidade do agente de superar obstáculos.

## Alteração principal

Foi adicionada uma matriz espacial 5×5 baseada no estado interno do
ambiente, representando:

- 0 = espaço vazio
- 1 = elemento sólido

A matriz foi mantida propositalmente pequena para evitar aumento
desnecessário da dimensionalidade das entradas.

## Configuração

- População: 50
- Gerações: 50
- Inputs anteriores: 37
- Inputs espaciais: 25
- Inputs totais: 62
- Outputs: 7
- Matriz espacial: 5×5
- Avaliação: paralela
- Workers: 12
- Seed do ambiente: 42
- Máximo de steps: 10.000
- Limite de estagnação: 180 steps
- Checkpoint: a cada 5 gerações

## Função de fitness

Mantida sem alterações:

fitness = progresso_horizontal
        + 10.000, se concluir a fase
        - 100, se morrer

A manutenção da mesma função de fitness permite avaliar de maneira mais
isolada o impacto da nova representação espacial.

## Resultado

- Melhor fitness: 1293
- Posição inicial X: 40
- Maior posição X: 1433
- Distância percorrida: 1393
- Steps executados: 638
- Reward acumulado do ambiente: 1345
- Fase concluída: Não
- Estado final: Morte

## Comportamento observado

Durante a avaliação visual do melhor indivíduo, foi observado que:

- o agente conseguiu superar os canos que limitavam os experimentos anteriores;
- desenvolveu comportamento de salto diante dos obstáculos;
- em determinado trecho realizou um salto diretamente de um cano para outro;
- conseguiu avançar significativamente além do ponto alcançado pelo EXP-002;
- posteriormente caiu em um buraco.

## Comparação com EXP-002

EXP-002:
Maior X = 900

EXP-003:
Maior X = 1433

Aumento:
533 unidades de posição X

Aumento relativo:
aproximadamente 59,2%

Fitness:
760 → 1293

## Interpretação

Os resultados fornecem evidência de que a representação espacial 5×5
melhorou a capacidade do agente de reagir à geometria do cenário.

A limitação predominante deixou de ser a incapacidade de superar canos.
O novo comportamento limitante observado foi a queda em um buraco.

Como a matriz espacial também representa a presença e ausência de solo,
não será adicionado imediatamente um sensor específico para buracos.
Primeiro será avaliado se um maior número de gerações permite ao próprio
processo evolutivo aprender a utilizar a informação espacial já disponível.

## Próximo passo

Executar treinamento prolongado mantendo:

- os mesmos 62 inputs;
- a mesma matriz 5×5;
- a mesma função de fitness;
- as mesmas ações;
- os mesmos parâmetros NEAT.

Objetivo:

Verificar se o aumento do número de gerações permite que o agente evolua
estratégias para superar buracos e avançar ainda mais na fase.


---

# EXP-003-PROLONGADO-150 — Treinamento prolongado da percepção espacial

## Objetivo

Avaliar se a mesma população evolutiva do EXP-003, mantendo
inalterados os sensores, a função de fitness e os hiperparâmetros
do NEAT, conseguiria desenvolver estratégias mais complexas apenas
com o aumento do número de gerações.

## Continuação do treinamento

O treinamento foi retomado a partir do checkpoint:

data/experiments/teste-003/checkpoints/neat-checkpoint-50

Dessa forma, a população, as espécies e o estado evolutivo obtidos
no EXP-003 foram preservados.

Não foi iniciada uma nova população aleatória.

## Configuração

- População: 50
- Geração inicial: 50
- Gerações adicionais: 150
- Geração final: 200
- Inputs: 62
- Outputs: 7
- Matriz espacial: 5×5
- Avaliação: paralela
- Workers: 10
- CPUs lógicas disponíveis: 12
- CPUs lógicas reservadas: 2
- Função de fitness: mantida sem alterações
- Parâmetros NEAT: mantidos sem alterações
- Checkpoint: a cada 5 gerações

## Hipótese

Aumentar o orçamento evolutivo, sem modificar a representação
sensorial nem a função de fitness, poderá permitir o surgimento de
novas estratégias capazes de superar os obstáculos que limitaram
o EXP-003 após 50 gerações.

## Resultado

- Geração inicial: 50
- Geração final: 200
- Gerações adicionais: 150
- Melhor fitness: 2186
- Tempo de treinamento: 1788,85 s
- Tempo aproximado: 29 min 49 s
- Workers: 10
- Fase concluída: Não

## Evolução observada

Durante o treinamento prolongado, o melhor fitness permaneceu
estagnado durante diversas gerações em aproximadamente 1295.

Apesar desse período de estagnação, novas variações genéticas
posteriormente conseguiram superar o melhor indivíduo anterior,
demonstrando que o processo evolutivo ainda possuía capacidade
de exploração.

O melhor agente final do treinamento prolongado conseguiu:

- superar os canos que limitavam os experimentos iniciais;
- superar o buraco que havia causado a morte do melhor agente
  do EXP-003;
- executar sequências de saltos mais complexas;
- avançar significativamente mais pela fase;
- alcançar uma região próxima ao final da fase.

## Novo comportamento limitante

Durante a avaliação visual do melhor indivíduo, foi identificado
um novo padrão de falha.

Após determinados saltos, o Mario podia aterrissar imediatamente
antes de uma parede, escada ou estrutura elevada.

Nessa situação, o agente permanecia executando ações de progressão
horizontal contra o obstáculo, sem conseguir desenvolver uma
estratégia adequada de recuperação.

O comportamento observado pode ser resumido como:

salto
→ aterrissagem muito próxima ao obstáculo
→ tentativa contínua de avançar
→ ausência de progresso horizontal
→ permanência no local

## Interpretação

O treinamento prolongado demonstrou que a representação espacial
5×5 possui capacidade suficiente para permitir a aprendizagem de
comportamentos mais complexos do que aqueles observados após apenas
50 gerações.

O agente conseguiu superar obstáculos que anteriormente limitavam
seu progresso, incluindo buracos e canos, sem qualquer alteração
manual nas regras de decisão.

Também foi observado que períodos prolongados sem melhoria do
melhor fitness não significam necessariamente que o processo
evolutivo tenha encerrado sua capacidade de aprendizagem.

Após diversas gerações sem superar o fitness anterior, uma nova
solução conseguiu romper o platô e avançar significativamente mais
pela fase.

## Comparação com EXP-003

EXP-003:
- Gerações totais: 50
- Melhor fitness: 1293
- Maior X: 1433
- Comportamento limitante: queda em buraco

EXP-003-prolongado-150:
- Gerações totais: 200
- Melhor fitness: 2186
- Comportamento limitante: bloqueio entre estruturas próximas
  ao final da fase

Variação de fitness:

1293 → 2186

Aumento absoluto:

893 pontos

Aumento relativo:

aproximadamente 69,1%

## Conclusão parcial

Os resultados indicam que a continuidade do processo evolutivo
permitiu ao agente desenvolver estratégias mais eficientes sem
necessidade de adicionar novos sensores ou modificar a função de
fitness.

Entretanto, o agente ainda apresenta limitações em situações de
recuperação após aterrissagens próximas a obstáculos elevados.

## Próximo passo

Será realizada uma nova continuação controlada da mesma população,
mantendo inalterados:

- os 62 inputs;
- a matriz espacial 5×5;
- os 7 outputs;
- a função de fitness;
- os parâmetros do NEAT;
- a lógica de ações;
- o ambiente de treinamento.

O objetivo será verificar se a população existente consegue evoluir
uma estratégia capaz de superar o novo padrão de falha observado.

Caso novas gerações não produzam melhoria significativa, será criado
um novo experimento independente com alterações controladas nos
hiperparâmetros do NEAT.


---

# EXP-004 — Ajuste de especiação

## Objetivo

Avaliar se uma configuração mais sensível de especiação poderia aumentar
a diversidade genética da população e favorecer o surgimento de novas
estratégias evolutivas.

O experimento foi motivado pelo fato de que, durante o EXP-003 e seu
treinamento prolongado, a população permaneceu concentrada em uma única
espécie durante praticamente todo o processo evolutivo.

## Alteração principal

Foram alterados apenas parâmetros relacionados à especiação e à
preservação de espécies.

Configuração anterior:

compatibility_threshold = 3.0
species_elitism = 2

Configuração utilizada no EXP-004:

compatibility_threshold = 2.5
species_elitism = 1

A redução do compatibility_threshold teve como objetivo tornar o processo
de especiação mais sensível às diferenças genéticas entre os indivíduos.

A redução de species_elitism teve como objetivo evitar a proteção
desnecessária de múltiplas espécies em uma população pequena.

## Configuração

- População: 50
- Gerações: 50
- Inputs: 62
- Outputs: 7
- Matriz espacial: 5×5
- Arquitetura: Feed-forward
- Função de ativação: tanh
- Conexões iniciais: full_direct
- Seed do ambiente: 42
- Avaliação: paralela
- Workers: 10
- CPUs lógicas disponíveis: 12
- CPUs lógicas reservadas: 2
- Máximo de steps: 10.000
- Limite de estagnação do episódio: 180 steps
- Checkpoint: a cada 5 gerações

## Função de fitness

Mantida sem alterações:

fitness = progresso_horizontal
        + 10.000, se concluir a fase
        - 100, se morrer

A manutenção dos sensores, da função de fitness e dos demais
hiperparâmetros permitiu avaliar de forma mais isolada o efeito das
alterações relacionadas à especiação.

## Hipótese

A redução do limiar de compatibilidade poderia favorecer a formação de
múltiplas espécies, preservando soluções geneticamente distintas e
aumentando a exploração do espaço de busca.

Esperava-se que uma maior diversidade pudesse reduzir períodos de
estagnação e aumentar a probabilidade de surgimento de estratégias
comportamentais diferentes.

## Resultado

- Geração inicial: 0
- Geração final: 50
- Melhor fitness: 1293
- Tempo de treinamento: 600,43 s
- Tempo aproximado: 10 minutos
- Workers utilizados: 10
- Número de espécies ao final: 1
- Fase concluída: Não

## Evolução observada

O treinamento apresentou um período inicial de evolução relativamente
lenta.

O melhor fitness iniciou em 858 e permaneceu próximo desse valor durante
diversas gerações.

Na geração 18 foi observado um pequeno avanço:

858 → 876

Após um novo período sem melhoria significativa, ocorreu um salto
expressivo na geração 41:

876 → 1278

Posteriormente, na geração 47, surgiu o melhor indivíduo do experimento:

1278 → 1293

O melhor fitness permaneceu em 1293 até o encerramento da geração 50.

O comportamento confirma novamente que períodos sem melhoria imediata
não significam necessariamente encerramento do processo evolutivo, uma
vez que novas soluções superiores podem surgir após várias gerações de
platô.

## Especiação observada

Apesar da alteração do compatibility_threshold de 3.0 para 2.5, a
população permaneceu agrupada em uma única espécie durante todo o
experimento.

Assim:

compatibility_threshold = 2.5

não foi suficiente, nesta configuração, para provocar a formação de
múltiplas espécies.

Foi observado aumento gradual da distância genética média entre os
indivíduos, porém essa diferença continuou dentro dos limites necessários
para agrupamento em uma única espécie.

Portanto, o principal objetivo estrutural do experimento, aumentar a
quantidade de espécies, não foi atingido.

## Comportamento observado

Durante a avaliação visual do melhor indivíduo, o comportamento aparentou
ser menos consistente do que o apresentado pelo agente obtido no
EXP-003-PROLONGADO-150.

Foram observados:

- saltos frequentes desde os primeiros trechos da fase;
- progressão inicial aparentemente menos controlada;
- superação de alguns obstáculos sem uma estratégia visualmente estável;
- queda posterior em um segundo buraco;
- ausência da consistência observada no melhor agente do treinamento
  prolongado do EXP-003.

Embora o fitness final tenha atingido 1293, o comportamento observado
sugere uma estratégia mais frágil e menos robusta.

## Comparação com EXP-003

EXP-003:

- Gerações: 50
- Inputs: 62
- compatibility_threshold: 3.0
- Melhor fitness: 1293
- Espécies: 1
- Comportamento: superou canos e caiu em um buraco

EXP-004:

- Gerações: 50
- Inputs: 62
- compatibility_threshold: 2.5
- Melhor fitness: 1293
- Espécies: 1
- Comportamento: progressão menos consistente e queda em buraco

Os dois experimentos alcançaram o mesmo melhor fitness após 50 gerações.

Entretanto, a alteração de especiação não produziu o aumento esperado na
quantidade de espécies e não apresentou benefício comportamental evidente.

## Comparação com EXP-003-PROLONGADO-150

EXP-003-PROLONGADO-150:

- Melhor fitness: 2186
- Gerações totais: 200
- Espécies: 1
- Superou canos e buracos
- Chegou próximo ao final da fase

EXP-004:

- Melhor fitness: 1293
- Gerações: 50
- Espécies: 1
- Caiu em um buraco
- Estratégia visualmente menos consistente

Essa comparação não deve ser utilizada como comparação direta de
eficiência entre configurações, pois os experimentos possuem diferentes
quantidades de gerações.

Entretanto, demonstra que a redução do compatibility_threshold para 2.5
não produziu uma vantagem evidente nas primeiras 50 gerações.

## Interpretação

O EXP-004 mostrou que pequenas alterações no limiar de compatibilidade
não necessariamente resultam em maior diversidade de espécies.

Mesmo com a redução de 3.0 para 2.5, todos os 50 indivíduos permaneceram
agrupados em uma única espécie.

Além disso, apesar de alcançar o mesmo melhor fitness do EXP-003 inicial,
o agente apresentou comportamento visualmente menos robusto.

O resultado também evidencia que o valor de fitness não deve ser analisado
isoladamente.

Dois agentes com fitness semelhante podem apresentar estratégias
comportamentais bastante diferentes.

Por esse motivo, a avaliação dos agentes deve considerar conjuntamente:

- fitness;
- distância alcançada;
- comportamento visual;
- consistência das ações;
- capacidade de superar diferentes tipos de obstáculos;
- quantidade de espécies;
- evolução ao longo das gerações.

## Conclusão parcial

A alteração:

compatibility_threshold: 3.0 → 2.5

não foi suficiente para aumentar a quantidade de espécies da população.

O EXP-004 também não apresentou melhoria comportamental evidente em
relação ao EXP-003 original.

Dessa forma, a configuração utilizada neste experimento não será
considerada superior à configuração anterior.

O resultado, entretanto, foi importante para demonstrar que pequenas
reduções no limiar de compatibilidade não garantem aumento da diversidade
e que maior diversidade genética deve ser avaliada empiricamente.

## Próximo passo

O próximo experimento deverá investigar uma alteração mais significativa
na especiação, mantendo novamente os sensores, a função de fitness e os
demais parâmetros constantes.

Uma possibilidade é reduzir adicionalmente o compatibility_threshold,
com o objetivo de verificar se um limiar mais baixo é capaz de produzir
múltiplas espécies.

A nova configuração deverá continuar sendo avaliada inicialmente com um
número controlado de gerações antes de qualquer treinamento prolongado.



EXP-005 — Ajuste mais agressivo de especiação

Objetivo

Avaliar se uma redução mais significativa do compatibility_threshold
seria capaz de produzir múltiplas espécies e aumentar a diversidade
evolutiva da população.

O EXP-004 demonstrou que a redução do limiar de compatibilidade de 3.0
para 2.5 não foi suficiente para separar a população em múltiplas espécies.
Por esse motivo, o EXP-005 realizou uma nova redução controlada do limiar,
mantendo os demais elementos principais do treinamento constantes.

Alteração principal

Configuração anterior no EXP-004:

compatibility_threshold = 2.5

species_elitism = 1

Configuração utilizada no EXP-005:

compatibility_threshold = 1.5

species_elitism = 1

A alteração teve como objetivo tornar o agrupamento por compatibilidade
genética mais restritivo, fazendo com que indivíduos geneticamente mais
distantes fossem separados em espécies distintas.

Configuração

População: 50

Gerações: 50

Inputs: 62

Outputs: 7

Matriz espacial: 5×5

Arquitetura: Feed-forward

Função de ativação: tanh

Conexões iniciais: full_direct

Seed do ambiente: 42

Avaliação: paralela

Workers: 10

CPUs lógicas disponíveis: 12

CPUs lógicas reservadas: 2

Máximo de steps: 10.000

Limite de estagnação do episódio: 180 steps

Checkpoint: a cada 5 gerações

Função de fitness

Mantida sem alterações:

fitness = progresso_horizontal
        + 10.000, se concluir a fase
        - 100, se morrer

A manutenção da representação sensorial, das ações e da função de fitness
permite avaliar de forma mais isolada o efeito da nova configuração de
especiação.

Hipótese

A redução do compatibility_threshold para 1.5 deverá aumentar a
sensibilidade do processo de especiação, permitindo que estruturas
geneticamente diferentes sejam preservadas em espécies distintas.

A maior diversidade poderá favorecer a exploração simultânea de diferentes
linhas evolutivas e aumentar a probabilidade de surgimento de estratégias
superiores.

Resultado

Geração inicial: 0

Geração final: 50

Melhor fitness: 1815

Tempo de treinamento: 540,73 s

Tempo aproximado: 9 min 1 s

Workers utilizados: 10

Maior quantidade observada de espécies: 4

Quantidade de espécies ao final: 3

Fase concluída: Não

Evolução do melhor fitness

Durante as 50 gerações, o melhor fitness apresentou os seguintes avanços
principais:

Geração 0   →  858
Geração 2   →  860
Geração 12  →  988
Geração 15  →  989
Geração 17  → 1381
Geração 28  → 1384
Geração 30  → 1661
Geração 48  → 1815

Diferentemente do EXP-004, o processo evolutivo não permaneceu concentrado
em uma única espécie.

O melhor indivíduo também não surgiu sempre da mesma linhagem. Ao longo
do treinamento, indivíduos pertencentes a diferentes espécies assumiram
a posição de melhor genoma da população.

Na geração 17, por exemplo, a espécie 4 produziu um indivíduo com fitness
1381. Posteriormente, na geração 30, a espécie 3 produziu um indivíduo com
fitness 1661. Finalmente, na geração 48, a espécie 2 produziu o melhor
indivíduo do experimento, com fitness 1815.

Esse comportamento indica que diferentes linhas evolutivas permaneceram
competitivas durante o treinamento.

Especiação observada

O objetivo estrutural do experimento foi atingido.

A população iniciou com duas espécies, passou para três e posteriormente
atingiu quatro espécies.

Durante o treinamento, espécies que permaneceram sem melhoria por muitas
gerações foram eliminadas pelo mecanismo de estagnação do NEAT.

Na geração 20, a espécie 1 foi removida após atingir o limite de
estagnação. Posteriormente surgiu uma nova espécie, identificada como
espécie 11.

Na geração 48, a espécie 4 também foi removida por estagnação.

Ao final do experimento permaneceram três espécies:

espécie 2;

espécie 3;

espécie 11.

Portanto, ao contrário do compatibility_threshold = 2.5 utilizado no
EXP-004, o valor 1.5 foi suficiente para produzir e preservar múltiplas
linhas evolutivas durante o treinamento.

Complexidade das redes

Também foi observado que o aumento do fitness não ocorreu simplesmente
por aumento contínuo da quantidade de nós ou conexões.

Exemplos de melhores indivíduos encontrados:

Geração 0  → fitness 858  → size (7, 434)
Geração 10 → fitness 860  → size (8, 416)
Geração 15 → fitness 989  → size (11, 417)
Geração 17 → fitness 1381 → size (7, 427)
Geração 30 → fitness 1661 → size (9, 409)
Geração 48 → fitness 1815 → size (9, 391)

O melhor indivíduo final apresentou menos conexões do que o indivíduo
inicial, reforçando que a evolução do NEAT não corresponde simplesmente
ao crescimento da rede. Diferentes topologias e combinações de pesos são
selecionadas de acordo com o desempenho obtido no ambiente.

Comportamento observado no viewer

Durante a avaliação visual do melhor indivíduo, o agente apresentou
progressão consistente pela fase e comportamento geral considerado
superior ao observado no EXP-004.

O agente conseguiu avançar bem pelo cenário e demonstrou capacidade de
reagir aos obstáculos utilizando a representação espacial disponível.

A execução terminou em morte antes da conclusão da fase. Entretanto,
visualmente, a falha final não evidenciou um padrão estrutural claro de
bloqueio, como o observado nos experimentos anteriores diante de canos,
buracos ou estruturas elevadas.

A morte aparentou decorrer de uma decisão pontual inadequada durante a
execução, e não de uma incapacidade evidente de interpretar determinado
tipo de obstáculo.

Essa distinção é importante: o resultado não permite afirmar que o agente
domina a fase, mas sugere que a configuração ainda possui potencial
evolutivo e merece ser avaliada com um orçamento maior de gerações.

Comparação com EXP-004

EXP-004:

compatibility_threshold = 2.5

Gerações: 50

Melhor fitness: 1293

Espécies ao final: 1

Comportamento visual: menos consistente

EXP-005:

compatibility_threshold = 1.5

Gerações: 50

Melhor fitness: 1815

Máximo observado: 4 espécies

Espécies ao final: 3

Comportamento visual: progressão mais consistente

Variação de fitness:

1293 → 1815

Aumento absoluto:

522 pontos

Aumento relativo:

aproximadamente 40,4%

Como ambos os experimentos utilizaram 50 gerações e mantiveram os demais
componentes principais constantes, essa comparação fornece evidência mais
direta de benefício da nova configuração do que uma comparação com o
EXP-003-PROLONGADO-150.

Ainda assim, uma única execução não é suficiente para concluir que o
limiar 1.5 é universalmente superior, devido à natureza estocástica do
processo evolutivo.

Comparação com EXP-003-PROLONGADO-150

O EXP-003-PROLONGADO-150 permanece com o maior fitness absoluto:

EXP-003-PROLONGADO-150: 2186 após 200 gerações totais

EXP-005: 1815 após 50 gerações

Essa comparação deve ser interpretada com cautela, pois os dois agentes
receberam orçamentos evolutivos muito diferentes.

O resultado relevante do EXP-005 é ter alcançado 1815 em apenas 50
gerações, além de apresentar diversidade de espécies que não existia no
treinamento prolongado anterior.

Por esse motivo, o EXP-005 torna-se um candidato relevante para treinamento
prolongado, permitindo verificar se a combinação entre maior diversidade
e maior orçamento evolutivo consegue superar o resultado de 2186 obtido
anteriormente.

Interpretação

O EXP-005 atingiu o objetivo de aumentar a diversidade de espécies.

A redução do compatibility_threshold para 1.5 modificou efetivamente a
dinâmica evolutiva: diferentes espécies coexistiram, produziram indivíduos
competitivos e foram submetidas ao mecanismo de estagnação e substituição.

Além da mudança estrutural, ocorreu melhoria quantitativa em relação ao
EXP-003 e ao EXP-004 após o mesmo número de gerações.

A avaliação visual também não revelou, até o momento, um obstáculo
específico que bloqueie sistematicamente o agente.

Os resultados, portanto, justificam investigar a continuidade dessa
população antes de introduzir novas alterações nos sensores, na função de
fitness ou nos demais hiperparâmetros.

Próximo passo

Realizar um treinamento prolongado a partir do checkpoint final do
EXP-005, sem alterar a configuração.

Devem ser mantidos:

os 62 inputs;

a matriz espacial 5×5;

os 7 outputs;

a função de fitness;

as ações disponíveis;

compatibility_threshold = 1.5;

species_elitism = 1;

os demais hiperparâmetros do NEAT;

o mesmo ambiente de treinamento.

O objetivo será verificar se a população que atingiu fitness 1815 em
50 gerações continua evoluindo quando recebe um orçamento maior de
gerações, especialmente se consegue superar o fitness 2186 do
EXP-003-PROLONGADO-150 e eventualmente concluir a fase.

A decisão sobre considerar essa configuração superior deverá levar em
conta não apenas o maior fitness, mas também o comportamento visual,
a consistência das ações, a capacidade de superar obstáculos e a
manutenção da diversidade evolutiva.


---

# EXP-005-PROLONGADO-150 — Treinamento prolongado com maior diversidade evolutiva

## Objetivo

Avaliar se a configuração do EXP-005, que apresentou maior diversidade
de espécies e melhor desempenho nas primeiras 50 gerações, conseguiria
continuar evoluindo quando submetida a um treinamento prolongado.

O treinamento foi retomado a partir do checkpoint final do EXP-005,
mantendo inalterados os sensores, a função de fitness, os parâmetros
do NEAT e a população evolutiva existente.

## Continuação do treinamento

Checkpoint utilizado:

data/experiments/teste-005/checkpoints/neat-checkpoint-50

A população não foi reiniciada.

Dessa forma, foram preservados:

- os genomas existentes;
- as espécies existentes;
- o estado evolutivo da população;
- os parâmetros utilizados no EXP-005.

## Configuração

- População: 50
- Geração inicial: 50
- Gerações adicionais: 150
- Geração final: 200
- Inputs: 62
- Outputs: 7
- Matriz espacial: 5×5
- compatibility_threshold: 1.5
- species_elitism: 1
- Avaliação: paralela
- Workers: 10
- Função de fitness: mantida sem alterações
- Parâmetros de mutação: mantidos sem alterações
- Checkpoint: a cada 5 gerações

## Hipótese

Como o EXP-005 apresentou maior diversidade de espécies e alcançou
fitness 1815 em apenas 50 gerações, esperava-se que um maior orçamento
evolutivo permitisse à população desenvolver estratégias ainda mais
eficientes e eventualmente superar o desempenho do
EXP-003-PROLONGADO-150.

## Resultado

- Geração inicial: 50
- Geração final: 200
- Gerações adicionais: 150
- Melhor fitness: 2186
- Tempo de treinamento: 1483,81 s
- Tempo aproximado: 24 min 44 s
- Workers utilizados: 10
- Fase concluída: Não

## Evolução observada

O treinamento prolongado permitiu que a população superasse o melhor
fitness obtido no EXP-005 original:

1815 → 2186

Aumento absoluto:

371 pontos

Aumento relativo:

aproximadamente 20,4%

O aumento demonstra que a população ainda possuía capacidade evolutiva
após as primeiras 50 gerações.

Entretanto, o melhor indivíduo final atingiu exatamente o mesmo fitness
máximo observado anteriormente no EXP-003-PROLONGADO-150.

## Comportamento observado no viewer

Durante a avaliação visual do melhor indivíduo, o agente apresentou
boa progressão pela fase e conseguiu superar os obstáculos iniciais.

No entanto, ao alcançar a região próxima ao final da fase, apresentou
novamente o mesmo padrão de falha observado no
EXP-003-PROLONGADO-150.

Após um salto, o Mario permaneceu posicionado entre estruturas elevadas
ou escadas e continuou executando ações de progressão horizontal sem
conseguir sair do local.

O comportamento observado foi:

salto
→ aterrissagem em posição desfavorável
→ obstáculo à frente
→ tentativa repetida de avançar
→ ausência de progresso horizontal
→ permanência no local

## Comparação com EXP-003-PROLONGADO-150

EXP-003-PROLONGADO-150:

- compatibility_threshold: 3.0
- Espécies predominantes: 1
- Gerações totais: 200
- Melhor fitness: 2186
- Comportamento limitante: bloqueio entre estruturas próximas ao final

EXP-005-PROLONGADO-150:

- compatibility_threshold: 1.5
- Múltiplas espécies durante a evolução
- Gerações totais: 200
- Melhor fitness: 2186
- Comportamento limitante: bloqueio no mesmo tipo de região

Apesar das diferenças na dinâmica de especiação, os dois treinamentos
prolongados atingiram o mesmo melhor fitness e apresentaram um padrão
de falha comportamental semelhante.

## Interpretação

O resultado sugere que a especiação não é mais o principal fator
limitante do agente nessa etapa.

No EXP-005, a redução do compatibility_threshold para 1.5 produziu
múltiplas espécies e acelerou a evolução inicial, permitindo alcançar
fitness 1815 em apenas 50 gerações.

Entretanto, após 200 gerações totais, o agente convergiu para o mesmo
fitness e para o mesmo padrão de falha observado em uma configuração
com apenas uma espécie.

Isso indica que o gargalo atual provavelmente está relacionado a outro
aspecto do agente.

A matriz espacial informa a geometria ao redor do Mario, porém a rede
não possui uma informação explícita sobre quanto tempo permanece sem
progresso.

Como a arquitetura é feed-forward, cada decisão depende apenas do estado
sensorial atual. Dessa forma, quando o Mario permanece parado diante de
um obstáculo, estados muito semelhantes continuam produzindo a mesma
ação repetidamente.

## Conclusão parcial

A maior diversidade evolutiva proporcionada pelo EXP-005 apresentou
benefício claro nas primeiras gerações, mas não foi suficiente para
eliminar o gargalo comportamental observado próximo ao final da fase.

O fato de duas configurações diferentes atingirem:

- o mesmo fitness máximo de 2186;
- o mesmo número total de gerações;
- e um padrão de falha visual semelhante;

fornece evidência de que simplesmente aumentar a diversidade ou prolongar
o treinamento pode não ser suficiente para resolver esse comportamento.

Por esse motivo, não será realizado imediatamente outro prolongamento
da mesma configuração.

## Próximo passo

O próximo experimento deverá introduzir uma informação temporal simples
sobre ausência de progresso.

Será adicionado um novo input denominado:

stagnation_ratio

Esse sensor representará, em escala normalizada entre 0 e 1, há quantos
steps consecutivos o Mario permanece sem aumentar sua maior posição
horizontal.

A intenção não é programar uma ação de recuperação manual.

O sensor apenas fornecerá à rede a informação de que o agente está
permanecendo sem progresso.

A decisão de:

- pular;
- mudar de direção;
- continuar avançando;
- ou executar outra ação;

continuará sendo determinada pela rede neural evoluída pelo NEAT.

O EXP-006 será executado inicialmente por 50 gerações, mantendo os demais
componentes constantes, permitindo avaliar de forma isolada o impacto
dessa nova informação temporal.



---

# EXP-006 — Adição do sensor de estagnação

## Objetivo

Avaliar se a inclusão de uma informação temporal explícita sobre ausência
de progresso horizontal permite ao agente reconhecer situações em que
permanece bloqueado e desenvolver estratégias alternativas de recuperação.

O experimento foi motivado pelo comportamento observado nos treinamentos
prolongados anteriores, nos quais agentes com bom desempenho conseguiam
avançar até regiões próximas ao final da fase, mas eventualmente permaneciam
pressionando contra obstáculos sem conseguir progredir.

## Alteração principal

Foi adicionado um novo sensor denominado `stagnation_ratio`.

Esse sensor representa a quantidade de steps consecutivos em que o Mario
permanece sem superar sua maior posição horizontal já alcançada.

O valor é normalizado no intervalo:

0.0 → progresso recente

1.0 → 180 ou mais steps sem novo progresso horizontal

Com isso, a quantidade total de entradas da rede passou de:

62 → 63 inputs

A intenção não foi programar uma ação específica para situações de
bloqueio, mas apenas fornecer à rede informação adicional para que o
próprio processo evolutivo pudesse descobrir como utilizá-la.

## Configuração

- População: 50
- Gerações: 50
- Inputs: 63
- Outputs: 7
- Arquitetura: Feed-forward
- Função de ativação: tanh
- Matriz espacial: 5×5
- Sensor de estagnação: Sim
- Limite de estagnação: 180 steps
- Avaliação: paralela
- Workers: 10
- Seed do ambiente: 42
- Máximo de steps: 10.000
- compatibility_threshold: 1.5
- species_elitism: 1
- Checkpoint: a cada 5 gerações

## Função de fitness

A função de fitness permaneceu inalterada:

fitness = progresso_horizontal
        + 10.000, se concluir a fase
        - 100, se morrer

A manutenção da função de fitness permitiu avaliar de forma mais isolada
o efeito da introdução do novo sensor.

## Resultado

- Geração inicial: 0
- Geração final: 50
- Melhor fitness: 1816
- Workers utilizados: 10
- Fase concluída: Não

## Evolução observada

O experimento apresentou aumento da diversidade genética em relação às
configurações iniciais.

Durante o treinamento foram observadas múltiplas espécies coexistindo,
demonstrando que o menor `compatibility_threshold` continuou promovendo
separação genética entre diferentes estratégias.

O melhor indivíduo atingiu fitness 1816, mas o processo apresentou
posteriormente um período prolongado sem superação significativa desse
valor.

## Comportamento observado

Durante a avaliação visual do melhor indivíduo:

- o agente apresentou boa progressão pela fase;
- conseguiu superar diferentes obstáculos;
- não chegou a demonstrar claramente o comportamento de recuperação
  esperado para situações de estagnação;
- morreu após avançar diretamente contra um inimigo.

A morte observada não permitiu concluir, apenas com as primeiras
50 gerações, se o novo sensor seria suficiente para solucionar o
problema de bloqueio identificado anteriormente.

## Interpretação

O sensor `stagnation_ratio` introduziu uma informação temporal que não
existia explicitamente nos 62 inputs anteriores.

Entretanto, fornecer uma informação à rede não significa que o processo
evolutivo necessariamente aprenderá a utilizá-la de forma eficiente.

Como o agente apresentou boa progressão e morreu por interação com um
inimigo antes de reproduzir o problema de bloqueio, decidiu-se ampliar
o orçamento evolutivo antes de avaliar definitivamente a utilidade
do novo sensor.

## Próximo passo

Continuar a população do EXP-006 por mais 150 gerações, sem modificar:

- sensores;
- função de fitness;
- parâmetros NEAT;
- ações disponíveis;
- ambiente de treinamento.

O objetivo será verificar se um maior número de gerações permite ao
processo evolutivo aprender a utilizar o `stagnation_ratio` e superar
o comportamento de bloqueio observado nos experimentos anteriores.


---

# EXP-006-PROLONGADO-150 — Treinamento prolongado com sensor de estagnação

## Objetivo

Avaliar se o aumento do orçamento evolutivo permite que a população
iniciada no EXP-006 aprenda a utilizar o sensor `stagnation_ratio`
para desenvolver estratégias de recuperação em situações de ausência
prolongada de progresso.

## Continuação do treinamento

O treinamento foi retomado a partir do checkpoint da geração 50 do
EXP-006.

Não foi criada uma nova população aleatória.

Foram preservados:

- população;
- espécies;
- genomas;
- estado evolutivo;
- configuração NEAT;
- representação sensorial.

## Configuração

- População: 50
- Geração inicial: 50
- Gerações adicionais: 150
- Geração final: 200
- Inputs: 63
- Outputs: 7
- Matriz espacial: 5×5
- Sensor de estagnação: Sim
- Avaliação: paralela
- Workers: 10
- CPUs lógicas disponíveis: 12
- CPUs lógicas reservadas: 2
- compatibility_threshold: 1.5
- species_elitism: 1
- Limite de estagnação: 180 steps
- Função de fitness: mantida sem alterações

## Resultado

- Geração inicial: 50
- Geração final: 200
- Gerações adicionais: 150
- Melhor fitness: 2188
- Fase concluída: Não
- Interrompido: Não

## Evolução observada

O treinamento prolongado permitiu superar o melhor fitness obtido nas
primeiras 50 gerações:

1816 → 2188

A população manteve diversidade genética durante o processo, chegando
a apresentar múltiplas espécies simultaneamente.

Entretanto, após alcançar o melhor fitness, ocorreu novamente um período
prolongado de platô evolutivo.

O melhor genoma permaneceu sem ser superado durante muitas gerações,
mesmo com a continuidade da exploração de outras espécies.

## Comportamento observado

Durante a avaliação visual do melhor indivíduo, foi identificado novamente
o mesmo comportamento limitante observado em treinamentos prolongados
anteriores.

O agente:

- avançou significativamente pela fase;
- superou canos e buracos;
- alcançou a região próxima ao final da fase;
- chegou à região das estruturas/escadas;
- permaneceu tentando avançar horizontalmente;
- não desenvolveu uma ação capaz de sair daquela situação;
- permaneceu estagnado no mesmo ponto.

## Resultado do sensor de estagnação

Apesar de o agente receber explicitamente informação sobre o tempo
decorrido sem novo progresso horizontal, o comportamento de bloqueio
continuou ocorrendo.

Portanto:

stagnation_ratio presente
        ↓
agente reconhece potencialmente o estado temporal
        ↓
processo evolutivo não desenvolve necessariamente
uma resposta comportamental útil

## Interpretação

O resultado demonstra que a simples disponibilização de uma nova
informação sensorial não garante que essa informação seja utilizada
de forma eficiente pela política evoluída.

O agente possuía informação espacial sobre o obstáculo e informação
temporal sobre ausência de progresso, mas ainda assim permaneceu
executando uma estratégia incapaz de superar aquela situação.

Isso sugere que o gargalo pode não estar exclusivamente na percepção.

Uma possível limitação encontra-se também na função de fitness.

A função utilizada até este experimento recompensa principalmente a
maior distância horizontal alcançada. Dessa forma, um agente que chega
muito longe e posteriormente permanece bloqueado ainda recebe uma
avaliação elevada.

Não existe uma pressão seletiva explícita diferenciando fortemente:

- chegar longe e permanecer bloqueado;
- chegar longe e continuar procurando uma estratégia de progressão.

## Conclusão parcial

O EXP-006-PROLONGADO-150 mostrou que:

- o aumento do treinamento elevou o melhor fitness;
- múltiplas espécies continuaram sendo exploradas;
- o agente alcançou novamente uma região avançada da fase;
- o `stagnation_ratio`, isoladamente, não solucionou o comportamento
  de bloqueio;
- simplesmente aumentar novamente o número de gerações não representa
  o próximo experimento mais informativo.

## Próximo passo

Modificar a função de fitness para introduzir uma pequena pressão seletiva
contra períodos prolongados sem progresso.

O próximo experimento deverá manter a arquitetura sensorial de 63 inputs,
mas diferenciar evolutivamente agentes que permanecem bloqueados daqueles
que mantêm comportamento de progressão.


---

# EXP-007 — Penalização leve por estagnação na função de fitness

## Objetivo

Avaliar se a introdução de uma penalização leve para períodos prolongados
sem progresso horizontal favorece a evolução de estratégias capazes de
evitar estados de bloqueio.

## Motivação

O EXP-006 demonstrou que informar à rede que o agente está estagnado não
foi suficiente para produzir uma estratégia de recuperação.

Por esse motivo, o EXP-007 manteve o sensor `stagnation_ratio`, mas
modificou a função de fitness para criar pressão seletiva contra períodos
prolongados sem progresso.

## Alteração principal

Foi introduzida uma tolerância inicial de 30 steps sem progresso.

Durante esse período nenhuma penalidade é aplicada, evitando punir
situações normais como:

- saltos;
- mudanças temporárias de direção;
- pequenas interrupções no movimento horizontal.

Após os 30 steps iniciais, cada step adicional sem superar o maior X
alcançado acrescenta uma pequena penalização acumulativa.

Parâmetros:

STAGNATION_GRACE_STEPS = 30

STAGNATION_PENALTY_PER_STEP = 0.05

O episódio continua sendo encerrado após:

MAX_STAGNATION_STEPS = 180

## Nova função de fitness

fitness = progresso_horizontal
        - penalidade_estagnacao
        + 10.000, se concluir a fase
        - 100, se morrer

Onde:

penalidade_estagnacao =
    steps_penalizados × 0.05

A penalização foi mantida propositalmente pequena para não substituir
o progresso horizontal como principal objetivo evolutivo.

## Configuração

- População: 50
- Gerações: 50
- Inputs: 63
- Outputs: 7
- Arquitetura: Feed-forward
- Matriz espacial: 5×5
- Sensor de estagnação: Sim
- Tolerância sem penalização: 30 steps
- Penalização por step: 0.05
- Limite máximo de estagnação: 180 steps
- compatibility_threshold: 1.5
- species_elitism: 1
- Avaliação: paralela
- Workers: 10
- Seed do ambiente: 42
- Máximo de steps: 10.000

## Resultado

- Geração inicial: 0
- Geração final: 50
- Melhor fitness: 1818
- Tempo de treinamento: 631,88 s
- Tempo aproximado: 10 min 32 s
- Workers utilizados: 10
- Interrompido: Não
- Fase concluída: Não

## Evolução observada

O melhor fitness de 1818 surgiu ainda nas primeiras gerações.

Posteriormente ocorreu um longo período sem superação do melhor indivíduo.

Apesar desse platô, a população apresentou diversidade genética
consideravelmente maior do que nos primeiros experimentos.

Foram observadas até aproximadamente seis espécies durante o processo,
e o treinamento terminou com cinco espécies coexistindo.

Esse comportamento demonstra que a especiação estava permitindo a
exploração simultânea de diferentes regiões do espaço genético, embora
nenhuma delas tenha conseguido superar significativamente o campeão
durante as 50 gerações.

## Comportamento observado

Durante a avaliação visual, o melhor indivíduo não apresentou o padrão
de bloqueio entre as escadas que havia caracterizado os treinamentos
prolongados anteriores.

O agente apresentou boa progressão, porém cometeu um erro de temporização:

- iniciou um salto antecipadamente;
- colidiu com um bloco;
- retornou ao chão;
- ficou posicionado diante de um Goomba;
- não conseguiu executar um novo salto a tempo;
- morreu devido à colisão com o inimigo.

## Interpretação

A falha observada foi qualitativamente diferente do comportamento de
estagnação investigado.

O agente não permaneceu indefinidamente pressionando contra um obstáculo.
A morte ocorreu devido a uma sequência específica de decisões e ao timing
do salto diante de um inimigo.

Dessa forma, não havia evidência suficiente para considerar a nova função
de fitness inadequada após somente 50 gerações.

Também é importante observar que os valores de fitness do EXP-007 não
devem ser comparados diretamente com os experimentos anteriores como se
pertencessem exatamente à mesma escala, pois a função de fitness passou
a incluir a penalização por estagnação.

## Próximo passo

Continuar o treinamento por mais 150 gerações, mantendo integralmente:

- a nova função de fitness;
- os 63 inputs;
- o sensor `stagnation_ratio`;
- a matriz espacial 5×5;
- os parâmetros NEAT;
- as ações disponíveis.

O objetivo será verificar se um orçamento evolutivo maior permite
refinar decisões de salto e evitar falhas pontuais observadas após
as primeiras 50 gerações.


---

# EXP-007-PROLONGADO-150 — Treinamento prolongado com penalização por estagnação

## Objetivo

Avaliar se a continuidade do processo evolutivo iniciado no EXP-007
permite aperfeiçoar a política obtida com a nova função de fitness,
reduzindo falhas de timing e mantendo a capacidade de evitar estados
prolongados de bloqueio.

## Continuação do treinamento

O treinamento foi retomado a partir da população existente no EXP-007.

Não foi criada uma nova população aleatória.

A função de fitness, sensores, ações e parâmetros NEAT foram mantidos
inalterados.

## Configuração

- População: 50
- Geração inicial: 50
- Gerações adicionais: 150
- Geração final: 200
- Inputs: 63
- Outputs: 7
- Arquitetura: Feed-forward
- Matriz espacial: 5×5
- Sensor de estagnação: Sim
- Tolerância de estagnação: 30 steps
- Penalização por step: 0.05
- Limite máximo de estagnação: 180 steps
- compatibility_threshold: 1.5
- species_elitism: 1
- Avaliação: paralela
- Workers: 10

## Resultado

- Geração inicial: 50
- Geração final: 200
- Gerações adicionais: 150
- Melhor fitness: 1883,55
- Tempo de treinamento: 1643,19 s
- Tempo aproximado: 27 min 23 s
- Workers utilizados: 10
- Fase concluída: Não

## Evolução observada

O treinamento prolongado conseguiu superar o melhor indivíduo das
primeiras 50 gerações:

1818 → 1883,55

Entretanto, o ganho foi relativamente pequeno diante das 150 gerações
adicionais.

Nas gerações finais, o genoma de fitness 1883,55 permaneceu como melhor
indivíduo, sem ser superado.

A população ainda apresentava quatro espécies nas gerações 198 e 199,
demonstrando que continuava existindo diversidade genética, apesar do
platô do melhor fitness.

## Comportamento observado

Na avaliação visual do melhor indivíduo, o agente novamente apresentou
uma falha relacionada à interação com um Goomba.

O comportamento observado foi:

- o agente avançou normalmente pela fase;
- realizou uma transição para o nível do chão;
- ao chegar ao solo, o Goomba já estava suficientemente próximo;
- não houve tempo disponível para iniciar um novo salto;
- o agente colidiu com o inimigo e morreu.

A falha não correspondeu ao antigo comportamento de permanecer
indefinidamente pressionando contra as estruturas próximas ao final
da fase.

## Interpretação

O treinamento prolongado mostrou que a nova função de fitness foi capaz
de produzir um agente com comportamento diferente daquele observado nos
experimentos que terminavam bloqueados entre estruturas.

Entretanto, 150 gerações adicionais produziram apenas uma melhoria
moderada no melhor fitness:

1818 → 1883,55

Além disso, o novo comportamento limitante está relacionado principalmente
à temporização das ações diante de inimigos e transições entre diferentes
alturas do cenário.

Não é possível concluir, a partir de uma única execução visual, que a
arquitetura feed-forward seja incapaz de solucionar esse problema.
Os sensores já fornecem informações relacionadas aos inimigos, incluindo
posição relativa e movimento.

Portanto, uma falha pontual de timing não constitui evidência suficiente
para justificar imediatamente a introdução de uma arquitetura recorrente.

## Conclusão parcial

O conjunto EXP-007 + EXP-007-PROLONGADO-150 mostrou que:

- a penalização por estagnação alterou a pressão seletiva;
- o comportamento limitante observado visualmente mudou;
- o melhor agente não reproduziu o bloqueio característico das escadas
  durante a avaliação realizada;
- o treinamento prolongado aumentou o fitness de 1818 para 1883,55;
- o processo entrou novamente em um platô;
- a falha final observada esteve associada ao timing diante de um inimigo.

O resultado indica que novos experimentos devem ser escolhidos com
cautela, evitando adicionar complexidade arquitetural apenas para
corrigir uma falha observada isoladamente.

## Próximo passo

Realizar um último experimento controlado antes da seleção da configuração
final.

Após esse experimento, deverá ser encerrada a etapa exploratória de
alterações sucessivas.

As melhores configurações deverão então ser comparadas e submetidas a:

- treinamentos independentes de maior duração;
- avaliação de consistência;
- análise da taxa de conclusão da fase;
- avaliação do melhor indivíduo;
- teste em outra fase sem retreinamento, para investigar capacidade
  de generalização.

---

# EXP-008 — Sensores explícitos de risco de aproximação de inimigos

## Objetivo

Avaliar se a inclusão de informações adicionais relacionadas à aproximação
de inimigos permitiria ao agente antecipar situações de colisão e melhorar
a temporização das ações de salto.

O experimento foi motivado pelas falhas observadas no EXP-007 e no
EXP-007-PROLONGADO-150, nas quais o agente apresentou boa progressão, mas
morreu em situações envolvendo Goombas e decisões de salto realizadas em
momentos inadequados.

## Alteração principal

A representação sensorial foi ampliada para fornecer informações adicionais
relacionadas ao risco de aproximação dos inimigos.

A quantidade de entradas da rede passou de:

63 → 68 inputs

Foram mantidos os sensores já existentes relacionados ao Mario, aos inimigos,
à percepção espacial 5×5 e à estagnação.

A alteração teve como objetivo fornecer à rede informações mais explícitas
sobre situações em que a distância entre Mario e um inimigo estava diminuindo,
sem programar diretamente qual ação deveria ser executada.

## Configuração

- População: 50
- Gerações: 50
- Inputs: 68
- Outputs: 7
- Arquitetura: Feed-forward
- Matriz espacial: 5×5
- Sensor de estagnação: Sim
- Sensores adicionais de aproximação de inimigos: Sim
- compatibility_threshold: 1.5
- species_elitism: 1
- Avaliação: paralela
- Workers: 10
- Seed do ambiente: 42
- Máximo de steps: 10.000
- Limite máximo de estagnação: 180 steps
- Penalização leve por estagnação: mantida

## Resultado

- Geração inicial: 0
- Geração final: 50
- Melhor fitness: 1635,50
- Fase concluída: Não

## Comportamento observado

Durante a avaliação visual do melhor indivíduo, o agente demonstrou capacidade
de reagir corretamente a alguns inimigos.

Foram observados momentos em que:

- identificou situações de aproximação;
- realizou saltos capazes de evitar alguns inimigos;
- continuou avançando pela fase.

Entretanto, esse comportamento não foi consistente durante toda a execução.

Em determinado momento, o agente continuou avançando diretamente contra outro
inimigo e morreu.

Portanto, a presença dos novos sensores não eliminou imediatamente as falhas
de interação com inimigos.

## Interpretação

O resultado indica que fornecer uma representação mais explícita do risco de
aproximação pode auxiliar determinadas decisões, mas não garante que a rede
desenvolva imediatamente uma política consistente para todas as situações.

Assim como ocorreu com o sensor de estagnação, a disponibilidade da informação
não implica necessariamente seu uso adequado pelo processo evolutivo.

Como o agente apresentou respostas corretas em algumas situações, considerou-se
prematuro descartar a nova representação após apenas 50 gerações.

## Próximo passo

Continuar a população do EXP-008 por mais 150 gerações, mantendo a mesma
configuração.

O objetivo será verificar se um orçamento evolutivo maior permite refinar o
uso das novas informações relacionadas aos inimigos e produzir uma política
mais consistente.


---

# EXP-008-PROLONGADO-150 — Treinamento prolongado com sensores de aproximação

## Objetivo

Avaliar se a continuidade do processo evolutivo iniciado no EXP-008 permite
aperfeiçoar o uso das informações adicionais relacionadas à aproximação de
inimigos.

## Continuação do treinamento

O treinamento foi retomado a partir do checkpoint final do EXP-008.

Não foi criada uma nova população aleatória.

Foram mantidos:

- os 68 inputs;
- a percepção espacial 5×5;
- o sensor de estagnação;
- os sensores adicionais relacionados aos inimigos;
- a função de fitness;
- as ações;
- os parâmetros do NEAT.

## Configuração

- População: 50
- Geração inicial: 50
- Gerações adicionais: 150
- Geração final: 200
- Inputs: 68
- Outputs: 7
- Arquitetura: Feed-forward
- compatibility_threshold: 1.5
- species_elitism: 1
- Avaliação: paralela
- Workers: 10

## Resultado

- Geração inicial: 50
- Geração final: 200
- Gerações adicionais: 150
- Melhor fitness: 2086,65
- Fase concluída: Não

## Comportamento observado

O treinamento prolongado aumentou consideravelmente o desempenho em relação
às primeiras 50 gerações:

1635,50 → 2086,65

Durante a avaliação visual, entretanto, o melhor agente ainda apresentou
comportamento pouco consistente.

O agente conseguiu avançar significativamente pela fase, porém acabou
permanecendo bloqueado ainda no início da região das estruturas/escadas.

Apesar da informação adicional sobre aproximação de inimigos, o treinamento
não solucionou de forma definitiva os principais problemas comportamentais.

## Interpretação

O aumento do fitness mostra que a população continuou evoluindo durante as
150 gerações adicionais.

Entretanto, a avaliação visual revelou novamente um comportamento de
estagnação semelhante aos encontrados anteriormente.

Isso reforça uma conclusão importante da sequência experimental: aumentar
progressivamente a quantidade de sensores não produz necessariamente uma
melhoria proporcional da política.

A rede passou de 37 entradas nos experimentos iniciais para 68 entradas nesta
configuração, mas ainda apresentou estratégias frágeis em determinadas
situações.

Portanto, novas ampliações sensoriais não devem ser realizadas sem uma
hipótese específica que as justifique.

## Conclusão parcial

O EXP-008-PROLONGADO-150 demonstrou que:

- os sensores adicionais permitiram algumas respostas adequadas a inimigos;
- o treinamento prolongado elevou o melhor fitness;
- o comportamento diante de inimigos ainda não se tornou completamente
  consistente;
- o agente voltou a apresentar estagnação na região das estruturas;
- simplesmente adicionar mais sensores não parece ser a solução principal
  para o gargalo atual.

## Próximo passo

Investigar a forma como o progresso é recompensado pela função de fitness.

Até este momento, grande parte da pressão evolutiva esteve associada à maior
posição horizontal atingida durante o episódio.

O próximo experimento deverá investigar uma recompensa de progresso mais
contínua, permitindo avaliar se fornecer feedback evolutivo durante o avanço
produz políticas diferentes.


---

# EXP-009 — Avaliação de uma nova estratégia de recompensa de progresso

## Objetivo

Avaliar o efeito de uma modificação na função de fitness destinada a fornecer
um sinal evolutivo mais contínuo durante a progressão pela fase.

O experimento foi realizado após os testes com percepção espacial, especiação,
sensor de estagnação, penalização por estagnação e informações adicionais
relacionadas aos inimigos.

## Configuração geral

- População: 50
- Gerações: 50
- Avaliação: paralela
- Workers: 10
- Seed do ambiente: 42
- Arquitetura: Feed-forward
- Representação sensorial da configuração experimental mais recente

## Resultado

- Geração inicial: 0
- Geração final: 50
- Melhor fitness: 1017,0
- Tempo de treinamento: 731,53 s
- Fase concluída: Não

## Evolução observada

O treinamento iniciou com um melhor fitness de aproximadamente 674,5.

Ao longo das gerações ocorreram melhorias graduais.

Na geração 17, o melhor fitness alcançou aproximadamente:

984,05

Posteriormente, na geração 29, surgiu o melhor indivíduo do experimento:

1017,0

Esse valor permaneceu sem ser superado até o encerramento da geração 50.

Portanto, a segunda metade do treinamento apresentou um platô considerável.

## Comportamento observado no viewer

A avaliação visual revelou uma limitação importante que não seria evidente
pela análise exclusiva do valor de fitness.

O agente conseguiu avançar por parte da fase, porém sua progressão aparentou
ser pouco consistente.

Ao se aproximar de um buraco, realizou o salto antecipadamente.

Como consequência:

salto antecipado
→ progressão temporária
→ posicionamento inadequado
→ queda no buraco
→ morte

Visualmente, parte do avanço obtido aparentou decorrer de uma sequência
favorável de ações, e não de uma estratégia robusta e repetível para superar
os obstáculos.

## Interpretação

O resultado reforça a necessidade de não utilizar apenas o maior fitness como
critério de seleção da configuração final.

Uma função que recompensa fortemente progresso durante o episódio pode
favorecer indivíduos que conseguem avançar, mesmo quando a sequência de ações
utilizada não representa uma política comportamental consistente.

O EXP-009 também apresentou um platô relativamente cedo: o melhor indivíduo
surgiu na geração 29 e não foi superado nas gerações restantes.

Diante da avaliação visual e da ausência de melhoria posterior significativa,
não há evidência suficiente para justificar imediatamente um treinamento
prolongado dessa população.

## Conclusão parcial

O EXP-009 mostrou que:

- modificar o sinal de recompensa altera significativamente a dinâmica
  evolutiva;
- progresso quantitativo não deve ser confundido automaticamente com
  comportamento inteligente ou robusto;
- a avaliação visual continua sendo necessária para interpretar o fitness;
- o melhor agente apresentou erro de temporização de salto e caiu em um
  buraco;
- o treinamento entrou em platô antes do encerramento das 50 gerações;
- não será realizado imediatamente um EXP-009-PROLONGADO-150.

## Próximo passo

Realizar uma última avaliação controlada da função de fitness antes de
encerrar a etapa exploratória.

Uma alternativa é utilizar uma função híbrida, preservando o maior progresso
horizontal como principal objetivo e adicionando apenas um componente menor
de recompensa incremental.

A estrutura conceitual pode ser representada como:

fitness =
    progresso_máximo
    + bônus_incremental_controlado
    - penalidade_de_estagnação
    - penalidade_de_morte
    + bônus_de_conclusão

O objetivo não será simplesmente aumentar o valor numérico do fitness, mas
verificar se a combinação produz comportamento mais consistente.

Após esse experimento, a etapa de alterações sucessivas deverá ser encerrada
e as configurações mais promissoras deverão ser selecionadas para avaliação
comparativa final.


---

# Evolução consolidada dos experimentos

| Experimento | Gerações | Inputs | Principal alteração | Melhor fitness | Resultado observado |
|---|---:|---:|---|---:|---|
| EXP-001 | 50 | 37 | Baseline | — | Bloqueado por cano |
| EXP-002 | 50 | 37 | Multiprocessing | 760 | Avançou até X=900 e permaneceu limitado pelo cano |
| EXP-003 | 50 | 62 | Matriz espacial 5×5 | 1293 | Superou canos e caiu em buraco |
| EXP-003-PROLONGADO-150 | 200 total | 62 | Continuação por +150 gerações | 2186 | Superou canos e buraco; chegou próximo ao final |
| EXP-004 | 50 | 62 | Ajuste de especiação para threshold 2.5 | 1293 | Continuou com uma espécie e apresentou comportamento menos consistente |
| EXP-005 | 50 | 62 | compatibility_threshold = 1.5 | 1815 | Múltiplas espécies e melhor evolução inicial |
| EXP-005-PROLONGADO-150 | 200 total | 62 | Continuação do EXP-005 | 2186 | Reproduziu bloqueio próximo às estruturas |
| EXP-006 | 50 | 63 | Sensor stagnation_ratio | 1816 | Boa progressão, mas morte diante de inimigo |
| EXP-006-PROLONGADO-150 | 200 total | 63 | Continuação do EXP-006 | 2188 | Voltou a estagnar na região das estruturas |
| EXP-007 | 50 | 63 | Penalização leve por estagnação | 1818 | Erro de timing diante de Goomba |
| EXP-007-PROLONGADO-150 | 200 total | 63 | Continuação do EXP-007 | 1883,55 | Falha de timing diante de inimigo |
| EXP-008 | 50 | 68 | Informações adicionais de aproximação de inimigos | 1635,50 | Desviou de alguns inimigos, mas morreu ao avançar contra outro |
| EXP-008-PROLONGADO-150 | 200 total | 68 | Continuação do EXP-008 | 2086,65 | Avançou mais, mas ficou bloqueado no início das estruturas |
| EXP-009 | 50 | — | Nova estratégia de recompensa de progresso | 1017,0 | Salto antecipado e queda em buraco; comportamento pouco consistente |


---

# ETAPA DE COMPARAÇÃO DOS EXPERIMENTOS

## Objetivo

Após a execução dos experimentos exploratórios, foi encerrada a etapa de
alterações sucessivas na arquitetura do agente.

A partir desse ponto, o objetivo deixou de ser adicionar novos sensores ou
modificar parâmetros imediatamente e passou a ser comparar quantitativamente
os resultados obtidos, buscando identificar quais configurações apresentavam
maior potencial para uma etapa posterior de treinamento prolongado.

A comparação foi dividida em duas etapas:

1. análise das métricas produzidas durante o treinamento;
2. avaliação padronizada dos melhores genomas (winners).

Essa separação foi adotada porque o maior valor de fitness isoladamente não é
suficiente para determinar qual agente apresenta o melhor comportamento,
principalmente porque diferentes experimentos utilizaram alterações na função
de fitness e na representação sensorial.


## Comparação das métricas de treinamento

Foi desenvolvido um script de análise responsável por percorrer os arquivos
`training_metrics.csv` dos experimentos e consolidar as informações produzidas
durante as gerações.

Foram considerados os seguintes indicadores:

- melhor fitness por geração;
- fitness médio da população;
- desvio padrão do fitness;
- quantidade de espécies;
- número de nós do melhor genoma;
- número de conexões do melhor genoma;
- tempo de execução das gerações.

A partir desses dados foram produzidos gráficos comparativos contendo:

- evolução do melhor fitness;
- evolução do fitness médio;
- evolução da quantidade de espécies;
- evolução da complexidade estrutural das redes.

Essa análise permitiu observar não apenas o maior indivíduo produzido, mas
também o comportamento da população ao longo das gerações, a ocorrência de
platôs evolutivos e as diferenças estruturais entre as configurações.

Os gráficos mostraram que diferentes experimentos apresentaram padrões
distintos de evolução e reforçaram que a escolha da configuração final não
deveria ser realizada somente a partir do maior valor numérico de fitness.


## Benchmark padronizado dos winners

Após a análise das curvas de treinamento, foi desenvolvido um segundo
procedimento destinado à avaliação dos melhores genomas de cada experimento.

Cada winner foi executado sob o mesmo protocolo de avaliação:

- ambiente: SuperMarioBros-1-1-v0;
- seed: 42;
- mesma lógica de execução da rede neural;
- mesma forma de seleção das ações;
- ausência de novo treinamento durante o benchmark.

A utilização de uma condição comum permitiu comparar indicadores independentes
das diferenças existentes entre as funções de fitness utilizadas durante os
experimentos.

Foram registrados:

- posição inicial no eixo X;
- maior posição X alcançada;
- distância horizontal máxima;
- quantidade de steps executados;
- reward acumulado fornecido pelo ambiente;
- conclusão da fase;
- ocorrência de morte;
- motivo de encerramento do episódio.


## Resultado do benchmark

Os resultados obtidos foram:

| Experimento | Inputs | Maior X | Distância | Steps | Reward | Concluiu | Morreu |
|---|---:|---:|---:|---:|---:|---|---|
| EXP-006-PROLONGADO-150 | 63 | 2228 | 2188 | 8019 | 1778 | Não | Sim |
| EXP-008-PROLONGADO-150 | 68 | 2147 | 2107 | 8019 | 1694 | Não | Sim |
| EXP-007-PROLONGADO-150 | 63 | 2027 | 1987 | 938 | 1927 | Não | Sim |
| EXP-010 | 68 | 1800 | 1760 | 622 | 1713 | Não | Sim |
| EXP-009 | 68 | 1153 | 1113 | 454 | 1075 | Não | Sim |

Nenhum dos agentes avaliados conseguiu concluir a fase durante o benchmark.

O EXP-006-PROLONGADO-150 apresentou a maior distância horizontal, alcançando
2188 unidades de progresso, seguido pelo EXP-008-PROLONGADO-150, com 2107
unidades.

O EXP-007-PROLONGADO-150 apresentou uma distância inferior, porém obteve
reward acumulado maior do que os dois primeiros. Esse resultado demonstrou
que reward do ambiente, fitness utilizado pelo processo evolutivo e distância
horizontal são métricas distintas e não devem ser interpretadas como valores
equivalentes.

Também foi observado que o EXP-010, apesar de apresentar desempenho
promissor durante sua evolução, não obteve a maior distância no benchmark
padronizado. Esse resultado reforça a importância da avaliação externa dos
winners antes da seleção de uma configuração para treinamento adicional.


## Investigação do encerramento dos EXP-006 e EXP-008

Durante o primeiro benchmark foi identificado que:

EXP-006-PROLONGADO-150:

- steps: 8019;

EXP-008-PROLONGADO-150:

- steps: 8019.

Como os dois agentes apresentaram exatamente a mesma quantidade de steps,
foi realizada uma análise adicional para determinar se esse valor correspondia
a algum limite artificial introduzido pelo código de benchmark.

O ambiente foi instrumentado para registrar:

- posição final;
- tempo restante no jogo;
- flag de conclusão;
- estado de morte;
- estado `terminated`;
- estado `truncated`.

Os resultados foram:

### EXP-006-PROLONGADO-150

- maior X: 2228;
- X final: 2226;
- distância: 2188;
- steps: 8019;
- tempo final: 0;
- flag_get: False;
- clear: False;
- death: True.

### EXP-008-PROLONGADO-150

- maior X: 2147;
- X final: 2146;
- distância: 2107;
- steps: 8019;
- tempo final: 0;
- flag_get: False;
- clear: False;
- death: True.

A investigação confirmou que os 8019 steps não foram causados por um limite
adicionado pelo script de benchmark.

Nos dois casos, o tempo interno da fase chegou a zero e o episódio terminou
em morte.

Também foi observado que a posição final permaneceu praticamente igual à
maior posição alcançada:

EXP-006:

2228 → 2226

EXP-008:

2147 → 2146

Portanto, os agentes não avançaram até uma determinada região e posteriormente
retornaram significativamente pelo cenário.

Eles permaneceram próximos ao ponto máximo alcançado até o término do tempo
da fase.


## Interpretação da comparação

A comparação quantitativa mostrou que o EXP-006-PROLONGADO-150 apresentou o
maior progresso horizontal entre os agentes avaliados.

O EXP-008-PROLONGADO-150 apresentou o segundo maior progresso, apesar de
utilizar uma representação sensorial mais ampla, composta por 68 entradas e
incluindo informações adicionais relacionadas ao risco de aproximação de
inimigos.

Nesse benchmark, portanto, a inclusão dos sensores adicionais do EXP-008 não
produziu desempenho horizontal superior ao EXP-006.

Esse resultado não permite concluir que os sensores adicionais sejam inúteis,
mas demonstra que sua inclusão não apresentou superioridade quantitativa neste
protocolo específico.

A comparação também revelou um comportamento limitante compartilhado pelos
dois melhores agentes: ambos conseguiram avançar por uma parcela significativa
da fase, porém permaneceram em uma região sem conseguir concluir o percurso
antes que o tempo disponível chegasse a zero.

Assim, o gargalo atual não pode ser atribuído simplesmente à incapacidade de
avançar pelos trechos iniciais da fase.

O problema experimental passa a estar relacionado à capacidade de continuar
progredindo após alcançar regiões mais avançadas do cenário.


## Conclusão da etapa comparativa

A análise conjunta das métricas de treinamento e do benchmark dos winners
permitiu encerrar a avaliação baseada exclusivamente em observações visuais.

Os resultados forneceram uma base quantitativa para a escolha dos próximos
experimentos.

Os principais resultados desta etapa foram:

- EXP-006-PROLONGADO-150 apresentou a maior distância;
- EXP-008-PROLONGADO-150 apresentou a segunda maior distância;
- nenhum winner concluiu a fase;
- EXP-006 e EXP-008 permaneceram ativos até o tempo interno da fase chegar a
  zero;
- aumentar a quantidade de informações sensoriais não produziu,
  automaticamente, maior progresso;
- fitness, reward e distância não devem ser utilizados como métricas
  equivalentes;
- a próxima alteração experimental deve ser fundamentada no comportamento
  limitante identificado nesta comparação, e não apenas na tentativa de
  aumentar o fitness.

## Próximo passo

Antes de iniciar um novo experimento ou um treinamento prolongado adicional,
deverá ser analisada a implementação atual da função de fitness e do processo
de avaliação utilizado nas configurações mais recentes.

O objetivo dessa análise será determinar exatamente quais comportamentos estão
sendo recompensados e penalizados atualmente.

Somente após essa verificação será definida a hipótese do próximo experimento,
evitando introduzir uma nova alteração sem uma justificativa diretamente
sustentada pelos resultados obtidos.