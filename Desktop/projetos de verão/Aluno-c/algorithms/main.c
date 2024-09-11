#include <stdio.h>
#include <stdlib.h>
#include  "biblioteca.h"
#include "biblioteca.c"


int main() {
Lista* minhaLista = cria_lista();
Aluno aluno1 = {20018030,"Vitor Hugo Amaro Aristides",10,10,8};
Aluno aluno2 = {20018040,"Lourdes Isabel",10,10,8};
Aluno aluno3 = {20018041,"Gleiton leonardo Amaro ",10,10,10};

printf("tamanho da lista ao iniciar: %d\n",tamanho_lista(minhaLista));
insere_lista_inicio(minhaLista,aluno1);
printf("Tamanho da lista ao adicionar um elemento: %d\n",tamanho_lista(minhaLista));
insere_lista_final(minhaLista,aluno2);
printf("tamanho da lista ao adicionar mais um elemento: %d\n",tamanho_lista(minhaLista));
insere_lista_ordenado(minhaLista,aluno3);
printf("tamanho da lista depois do Ordenado:%d",tamanho_lista(minhaLista));
};
