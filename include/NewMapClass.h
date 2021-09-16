#ifndef NEW_MAP_H
#define NEW_MAP_H
#pragma once

#include <map>
#include <mutex>

template <class T1, class T2>
class NewMapClass {
public:
	NewMapClass();
	virtual ~NewMapClass();
private:
	std::mutex mMutex;
	std::map<T1, T2> mMap;
public:
	size_t Count(T1 id);
	void Update(T1 id, T2 data);
	T2   Get(T1 id);
	size_t Size();
};

template <class T1, class T2>
NewMapClass<T1, T2>::NewMapClass() {}
template <class T1, class T2>
NewMapClass<T1, T2>::~NewMapClass() {}
template <class T1, class T2>
size_t NewMapClass<T1, T2>::Count(T1 id) {
	std::unique_lock<std::mutex> lock(mMutex);
	return mMap.count(id);
	//return mMap.count(id) ? true : false;
}
template <class T1, class T2>
void NewMapClass<T1, T2>::Update(T1 id, T2 data) {
	std::unique_lock<std::mutex> lock(mMutex);
	mMap[id] = data;
}
template <class T1, class T2>
T2   NewMapClass<T1, T2>::Get(T1 id) {
	std::unique_lock<std::mutex> lock(mMutex);
	return mMap[id];
}
template <class T1, class T2>
size_t NewMapClass<T1, T2>::Size() {
	std::unique_lock<std::mutex> lock(mMutex);
	return mMap.size();
}

#endif