#include  "biblioteca.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct aluno{
    long matricula;
    char nome[30];
    float n1,n2,n3;

};
// Estrutura lista
struct lista{
    int qtd;
    struct aluno dados[MAX];
};

typedef struct lista Lista;
typedef struct aluno Aluno;


// função para criar lista
Lista* cria_lista(){
    Lista *li;
    li =(Lista*)malloc(sizeof(struct lista));
    if(li != NULL){
         li-> qtd = 0;
        printf("\nLista criada!\n");
    }return li;
    };
// função para apagar lista
void libera_lista(Lista* li){
    free(li);
};

//Função tamanho da lista
int tamanho_lista(Lista* li){
      if (li == NULL) {
        return -1;
    } else {
        return li->qtd;     // Retorna o valor de 'qtd' da lista
    }
};

//Lista cheia
int lista_cheia(Lista* li){
    if(li == NULL){
        return -1;
    }return(li->qtd == MAX);
}

//Lista vazia
int lista_vazia(Lista* li){
    if(li == NULL){
        return -1;
    }return(li->qtd==0);
}

//inserção no inicío 
// ? a cada inserção muda o lugar dos elementos da lista uma posição a frente
int insere_lista_inicio(Lista* li, struct aluno al){
    if(li == NULL)//* checando se a lista foi criada
        return 0;
    if(li->qtd == MAX)//* checando se o valor maximo da lista já foi atingida
        return 0;
    int i;
    for(i =li->qtd-1; i>=0; i--)
        li->dados[i+1]= li->dados[i];
    li->dados[0]= al;
    li->qtd++;
    return 1;
}

//insere no final
int insere_lista_final(Lista* li, struct aluno al){
    if(li ==NULL)
        return -1;
    if(li->qtd == MAX)
        return -1;
    li->dados[li->qtd] = al;
    li->qtd++;
    return 1;
}

//Insere de maneira ordenada
//? caso iseira no começo ou no final será preciso deslocar os outros elementos da lista
int insere_lista_ordenado(Lista* li, struct aluno al){
    if(li == NULL){
        printf("Erro ao criar a lista ");
        return 0;
    };
    if(li-> qtd == MAX){
        printf("Lista sta cheia");
        return 0;
    };
    int k,i = 0;
    while(i< li->qtd && li->dados[i].matricula<al.matricula)
        i++;

    for(k=li->qtd-1;k>=i;k--){
        li->dados[k+1] = li->dados[k];
    };
    li->dados[i] = al;
    li->qtd++;
    return 1;

};
