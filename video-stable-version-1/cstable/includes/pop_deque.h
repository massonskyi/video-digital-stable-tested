#ifndef POP_DEQUE_H
#define POP_DEQUE_H

#include <deque>
#include <stdexcept>
#include <limits>

template <typename T>
class PopDeque : public std::deque<T> {
public:
    PopDeque(size_t maxlen = std::numeric_limits<size_t>::max()) : maxlen(maxlen) {}

    bool deque_full() const {
        return this->size() == maxlen;
    }

    T pop_append(const T& x) {
        T popped_element;
        if (deque_full()) {
            popped_element = this->front();
            this->pop_front();
        }

        this->push_back(x);

        return popped_element;
    }

    T increment_append(T increment = 1, bool pop_append = true) {
        T popped_element;
        if (this->empty()) {
            popped_element = pop_append(0);
        } else {
            popped_element = pop_append(this->back() + increment);
        }

        if (!pop_append) {
            return T();
        }

        return popped_element;
    }

    T pop_front() {
        if (this->empty()) {
            throw std::out_of_range("PopDeque is empty");
        }
        T front = this->front();
        this->pop_front();
        return front;
    }

private:
    size_t maxlen;
};

#endif // POP_DEQUE_H