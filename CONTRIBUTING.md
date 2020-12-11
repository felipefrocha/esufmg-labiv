# Contribuições
---
## Documentação
Somente edite a pasta de [documentação](/doc) por meio do [overleaf](https://www.overleaf.com/read/jdhgpcfvzxzt)

## CICD 
Não rode analises desnecessárias

## Code Review
Sempre peça que alguém revise seu código quando abrir um [PR](https://github.com/features/code-review/)

## Git(hub) Flow
Tenha sempre em mente o fluxo de trabalho que irá realizar, no caso estamos seguindo o [GitHub Flow](https://guides.github.com/introduction/flow/)
e nunca deixe de repassar seus commits e caso seja necessário eliminar alguns veja como fazer:

```bash
git rebase -i HEAD~<numero de commits a apagar>
```
Ao executar o comando acima ira aparecer uma seleção de commits (numero que você solicitou) ponha s nos commits que deseja fundir no primeiro commit da lista.
lembre de deixar o primeiro commit intacto.
