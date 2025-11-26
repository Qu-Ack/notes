= Introduction

Network Layer Provides host-to-host communication, it's the most interesting layer and will learn how it implements host-to-host communication.


+ Data Plane
  - primary role of each dataplane is to forward datagrams from it's input link to it's output link.

+ Control Plane
  - primary role is to control these per-router forwarding actions so that datagrams are ultimately transfered end to end.




= Routing Vs Forwarding

Forwarding is the process forwarding a packet at router's input interface to it's output interface. Forwarding usually takes nanoseconds to perform

Routing is the process of thinking about the path the packet should take to reach the destination address. Routing takes seconds. to perform.

Think of this analogy. a driver driving from delhi to mumbai may pass many toll booths, forwarding is like passing those toll boths. Routing is like planning the journey from delhi to mumbai, the ideal time we should start our journey, the highways we should take and all that stuff.






