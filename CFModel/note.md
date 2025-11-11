result -> mmr rerank # bat buoc vi nhieu item giong nhau -> ko co exploration
content base = ann research dua tren user vector
CF = model

user vector:
    location
    seniority
    salary min max
    currency
    remote pref
    relocation pref (recommend nhiều khu vực)
    skills = candidate base skills + extract(1-n cv candidate upload)
    education: need cause mot so cong ty sv yeu cau HCMC thay vi blalal T.T -> embedding them


sửa lại sync processor khi embedding job: tách ra từng phần rồi weight để phân biệt được đâu quan trọng trong embedding