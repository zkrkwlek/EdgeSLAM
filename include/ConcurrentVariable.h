#ifndef CONCURRENT_VARIABLE_H
#define CONCURRENT_VARIABLE_H
#pragma once

#include <mutex>

template <class T>
class ConcurrentVariable {
public:
	ConcurrentVariable();
	virtual ~ConcurrentVariable();
private:
	std::mutex mMutex;
	T variable;
public:
	void Update(T data);
	T   Get();
};

template <class T>
ConcurrentVariable<T>::ConcurrentVariable() {}
template <class T>
ConcurrentVariable<T>::~ConcurrentVariable() {}

template <class T>
void ConcurrentVariable<T>::Update(T data) {
	std::unique_lock<std::mutex> lock(mMutex);
	variable = data;
}
template <class T>
T ConcurrentVariable<T>::Get() {
	std::unique_lock<std::mutex> lock(mMutex);
	return variable;
}
#endif