#ifndef CONCURRENT_VECTOR_H
#define CONCURRENT_VECTOR_H
#pragma once

#include <mutex>

template <class T>
class ConcurrentVector {
public:
	ConcurrentVector();
	virtual ~ConcurrentVector();
private:
	std::mutex mMutex;
	std::vector<T> mVector;
public:
	void Initialize(int N, T data);
	void push_back(T data);
	T get(int idx);
	std::vector<T> get();
	void update(int idx,T data);
	size_t size();
};

template <class T>
ConcurrentVector<T>::ConcurrentVector() {}
template <class T>
ConcurrentVector<T>::~ConcurrentVector() {}

template <class T>
void ConcurrentVector<T>::Initialize(int N, T data) {
	std::unique_lock<std::mutex> lock(mMutex);
	mVector = std::vector<T>(N, data);
}

template <class T>
void ConcurrentVector<T>::push_back(T data) {
	std::unique_lock<std::mutex> lock(mMutex);
	mVector.push_back(data);
}

template <class T>
T ConcurrentVector<T>::get(int idx) {
	std::unique_lock<std::mutex> lock(mMutex);
	return mVector[idx];
}

template <class T>
std::vector<T> ConcurrentVector<T>::get(){
	std::unique_lock<std::mutex> lock(mMutex);
	return std::vector<T>(mVector.begin(), mVector.end());
}

template <class T>
void ConcurrentVector<T>::update(int idx, T data) {
	std::unique_lock<std::mutex> lock(mMutex);
	mVector[idx] = data;
}

template <class T>
size_t ConcurrentVector<T>::size() {
	std::unique_lock<std::mutex> lock(mMutex);
	return mVector.size();
}

#endif