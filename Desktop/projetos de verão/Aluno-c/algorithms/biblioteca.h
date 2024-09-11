#ifndef BIBLIOTECA_H
#define BIBLIOTECA_H
#define MAX 100


typedef struct aluno Aluno;
typedef struct lista Lista;


Lista* cria_lista();
void libera_lista(Lista* li);
int busca_lista_pos(Lista* li, int pos, struct aluno *al);
int busca_lista_met(Lista* li, int mat, struct aluno *al);
int insere_lista_final(Lista* li, struct aluno);
int insere_lista_inicio(Lista* li, struct aluno al);
int insere_lista_ordenado(Lista* li,struct aluno al);
int remove_lista_(Lista* li,int mat);
int remove_lista_inicio(Lista* li);
int remove_lista_final(Lista* li);
int tamanho_lista(Lista* li);
int lista_cheia(Lista* li);
int lista_vazia(Lista* li);

#endif