# Lesson 3. The Transport Layer: TCP and UDP

Sample lecture written by the repository author for demo purposes, released under CC BY 4.0.

## 3.1 Role of the transport layer

The transport layer provides a logical communication channel between processes running on two different hosts. While the network layer moves packets between two hosts, the transport layer moves data between two processes. Each process is identified by an IP address combined with a port number. A port number is a 16-bit integer ranging from 0 to 65535. Ports 0 to 1023 are called well-known ports and are reserved for common services. For example, HTTP uses port 80, HTTPS uses port 443 and SSH uses port 22.

## 3.2 The UDP protocol

UDP is a connectionless protocol: the sender does not need to set up a connection before sending data. UDP does not guarantee delivery, does not guarantee ordering and has no congestion control. In exchange, the UDP header is only 8 bytes long, so processing overhead is very low. UDP suits applications that need low latency and can tolerate some data loss, such as voice calls, live streaming and online games. The Domain Name System (DNS) also usually runs over UDP because each query consists of one short request and one short response.

## 3.3 TCP and the three-way handshake

TCP is a connection-oriented, reliable protocol. Before any data is transferred, the two sides perform a three-way handshake. In step one, the client sends a SYN segment carrying its initial sequence number. In step two, a listening server replies with a SYN-ACK segment, which acknowledges the client's sequence number and carries the server's own initial sequence number. In step three, the client sends an ACK to confirm. The three-way handshake lets both sides synchronize their initial sequence numbers and confirm that both directions of the connection work. If the client sends a SYN but never receives a SYN-ACK, the usual cause is that no server process is listening on that port or a firewall has dropped the packet.

## 3.4 Reliable delivery in TCP

TCP numbers every byte of data. The receiver sends back an acknowledgement number equal to the sequence number of the next byte it expects. If the sender does not receive an acknowledgement before its timer expires, it retransmits that segment. When the sender receives three duplicate acknowledgements for the same sequence number, it infers that a segment was lost and retransmits it immediately without waiting for the timeout; this mechanism is called fast retransmit.

## 3.5 Flow control and congestion control

Flow control prevents the sender from overflowing the receiver's buffer. The receiver advertises its receive window size in every segment, and the sender may not have more unacknowledged data in flight than this window. Congestion control, in contrast, protects the network rather than the receiver. During slow start, the congestion window doubles every round-trip time. Once the window exceeds a threshold, TCP switches to congestion avoidance and grows the window by only one segment per round-trip time. When packet loss is detected, TCP shrinks the congestion window to reduce the load on the network.

## 3.6 Closing a TCP connection

To close a connection, each side sends a FIN segment and waits for the other side to acknowledge it with an ACK, so closing usually involves four segments. The side that actively closes the connection stays in the TIME_WAIT state for a while to make sure the final ACK arrives and that old segments still in the network are not mistaken for a new connection.
