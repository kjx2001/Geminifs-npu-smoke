#include "list.h"
#include <linux/types.h>
#include <linux/spinlock.h>
#include <linux/printk.h>
#include <asm/errno.h>
#include <linux/compiler.h>

//  head 自环：head.prev = head.next = &head；初始化自旋锁
void list_init(struct list *list)
{
    list->head.list = list;
    list->head.prev = &list->head;
    list->head.next = &list->head;

    spin_lock_init(&list->lock);
}

// 摘除：prev->next=next; next->prev=prev; 然后把 e 的指针清 NULL
void list_remove(struct list_node *element)
{
    if (likely(element != NULL && element->list != NULL && element != &element->list->head))
    {
        spin_lock(&element->list->lock);
        element->prev->next = element->next;
        element->next->prev = element->prev;
        spin_unlock(&element->list->lock);

        element->list = NULL;
        element->next = NULL;
        element->prev = NULL;
    }
}

// 尾插：插到 head.prev 之后（加 spinlock 保护）
void list_insert(struct list *list, struct list_node *element)
{
    struct list_node *last = NULL;

    spin_lock(&list->lock);
    last = list->head.prev;
    last->next = element;

    element->list = list;
    element->prev = last;
    element->next = &list->head;

    list->head.prev = element;

    spin_unlock(&list->lock);
}
