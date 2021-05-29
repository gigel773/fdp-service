project(nips CXX)

find_package(OpenSSL REQUIRED)
find_package(cpprestsdk REQUIRED)
find_package(Boost COMPONENTS thread log REQUIRED)
find_package(OpenCV REQUIRED)

add_library(nips src/eureka_client.cpp src/nips.cpp)
target_include_directories(nips PUBLIC include PRIVATE src)
target_link_libraries(nips PRIVATE cpprestsdk::cpprest ${Boost_LIBRARIES} OpenSSL::Crypto nip)
