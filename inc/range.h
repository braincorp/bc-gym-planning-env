
#pragma once
/*
https://stackoverflow.com/questions/7185437/is-there-a-range-class-in-c11-for-use-with-range-based-for-loops
Allows you write more concise new style for loops with integers
for(auto i : range(2, 6)) std::cout << i << std::endl;
for(auto i : range(array.shape()[0])) std::cout << i << std::endl;
*/
class range {
 public:
   range(long int begin, long int end) : begin_(begin), end_(end) {}
   range(long int end) : range(0, end) {}

   class iterator {
      friend class range;
    public:
      long int operator *() const { return i_; }
      const iterator &operator ++() { ++i_; return *this; }
      iterator operator ++(int) { iterator copy(*this); ++i_; return copy; }

      bool operator ==(const iterator &other) const { return i_ == other.i_; }
      bool operator !=(const iterator &other) const { return i_ != other.i_; }

    protected:
      iterator(long int start) : i_ (start) { }

    private:
      unsigned long i_;
   };

   iterator begin() const { return begin_; }
   iterator end() const { return end_; }
private:
   iterator begin_;
   iterator end_;
};

/*
A macro to suppress unused var gcc complaints
https://stackoverflow.com/questions/21792347/unnamed-loop-variable-in-range-based-for-loop/21800058#21800058
*/
#if defined(__GNUC__)
#  define UNUSED __attribute__ ((unused))
#elif defined(_MSC_VER)
#  define UNUSED __pragma(warning(suppress:4100))
#else
#  define UNUSED
#endif
