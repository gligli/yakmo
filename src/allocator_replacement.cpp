#include "oneapi/tbb/scalable_allocator.h"

void* calloc(size_t num, size_t size)
{
  return scalable_calloc(num, size);
}

void free(void* memptr)
{
  scalable_free(memptr);
}

void* malloc(size_t memSize)
{
  return scalable_malloc(memSize);
}

void* realloc(void* memptr, size_t memSize)
{
  return scalable_realloc(memptr, memSize);
}

//void* _expand(void* memptr, size_t newSize)
//{
//  return nullptr;
//}

//size_t _msize(void* memptr)
//{
//  return scalable_msize(memptr);
//}

void* operator new(size_t size)
{
  void* ptr = scalable_malloc(size);
  if (ptr)
    return ptr;
  else
    throw std::bad_alloc();
}

void* operator new[](size_t size)
{
  void* ptr = scalable_malloc(size);
  if (ptr)
    return ptr;
  else
    throw std::bad_alloc();
}

void* operator new(size_t size, const std::nothrow_t&) throw()
{
  return scalable_malloc(size);
}

void* operator new[](size_t size, const std::nothrow_t&) throw()
{
  return scalable_malloc(size);
}

void operator delete(void* pointer) throw()
{
  scalable_free(pointer);
}

void operator delete[](void* pointer) throw()
{
  scalable_free(pointer);
}

void operator delete(void* pointer, const std::nothrow_t&) throw()
{
  scalable_free(pointer);
}

void operator delete[](void* pointer, const std::nothrow_t&) throw()
{
  scalable_free(pointer);
}

